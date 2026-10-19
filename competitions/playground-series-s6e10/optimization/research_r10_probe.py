"""Validation-device and checkpoint test, then full-size random-label resource test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,time,threading,subprocess,resource,gc,cloudpickle
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT,now,dump,sha
from research_r7_loader import load_split
from research_r4_lean_features import LeanFeatures
from research_r8_probe import BoundedRealMLP,CAP,CEILING
from research_r10_streaming import validation_streaming_backend,ValidationStreamingModule
from pytabkit.models.training.metrics import Metrics

def main():
    full='--full' in os.sys.argv
    path=ROOT/('research_r10_full_probe.json' if full else 'research_r10_compatibility.json')
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rec=dict(status='running',started_utc=now(),official_outer_quality_scored=False,
             runner_sha256=sha(__file__),adapter_sha256=sha(ROOT/'research_r10_streaming.py'))
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:
                        dump(path,dict(rec,status='resource_guard_stopped',peak_gpu_process_bytes=peak[0]));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        df,raw,y,splits,*_=load_split('oof');fi=splits[0][0]
        if not full:fi=fi[:24000]
        # Random labels exercise native AUC/checkpoint plumbing without selecting
        # a recipe using any official outer validation target or quality score.
        rng=np.random.default_rng(20261027);labels=rng.integers(0,2,len(fi),dtype='int8')
        it,iv=np.arange(int(.9*len(fi))),np.arange(int(.9*len(fi)),len(fi))
        tr=LeanFeatures();X=tr.fit_transform(raw.iloc[fi[it]].reset_index(drop=True),labels[it]);V=tr.transform(raw.iloc[fi[iv]].reset_index(drop=True))
        cfg=dict(json.loads((ROOT/'research_r8_plan.json').read_text())['config']);cfg.pop('stop_epoch');cfg.update(n_epochs=1 if full else 3,n_ens=8 if full else 2,val_metric_name='1-auc_ovr')
        rec.update(config=cfg,fit_rows=len(it),validation_rows=len(iv),epochs=[])
        tick=time.monotonic();original=ValidationStreamingModule.on_validation_epoch_end
        snapshots={};replay_batches=[]
        def val_hook(module):
            assert str(module.val_dl.val_x_dl.ds.device)=='cpu' and module.val_dl.val_y.device.type=='cuda'
            assert module.val_preds and all(p.device.type=='cuda' for p in module.val_preds)
            original(module)
            if not full:
                snapshots[int(module.current_epoch)+1]={k:v.detach().cpu().clone() for k,v in module.model.state_dict().items()}
                if not replay_batches:
                    replay_batches.extend({k:v.detach().cpu().clone() for k,v in batch.items()} for batch in module.val_dl)
            rec['epochs'].append(dict(epoch=int(module.current_epoch)+1,validation_data_device='cpu',validation_labels_device='cuda:0',model_devices=sorted({str(p.device) for p in module.parameters()})))
        ValidationStreamingModule.on_validation_epoch_end=val_hook
        try:
            with validation_streaming_backend(),threadpool_limits(6):model=BoundedRealMLP(**cfg).fit(X,labels[it],X_val=V,y_val=labels[iv])
        finally:ValidationStreamingModule.on_validation_epoch_end=original
        module=model.cv_alg_interface_.model
        selection=model.cv_alg_interface_.fit_params[0]['stop_epoch']['1-auc_ovr']
        assert 1<=selection<=cfg['n_epochs'] and len(rec['epochs'])==cfg['n_epochs']
        if not full:
            # Replay the same CUDA validation path and verify the exact selected
            # state. The public API performs ensemble softmax on CPU instead,
            # which changes ties for these intentionally near-.5 random scores.
            state=module.model.state_dict()
            state_error=max(float((state[k].detach().cpu()-value).abs().max()) for k,value in snapshots[int(selection)].items())
            assert state_error<1e-7
            rec['selected_checkpoint_state_max_error']=state_error
            module.to('cuda:0');module.model.eval()
            with torch.no_grad():
                logits=torch.cat([module.model({k:v.to('cuda:0') for k,v in batch.items()})['x_cont'] for batch in replay_batches],dim=-2)
                logits=module._postprocess_ens_pred(logits)
                replay=float(Metrics.apply(logits[0],torch.as_tensor(labels[iv],device='cuda:0')[:,None],'1-auc_ovr'))
            recorded=float(module.best_mean_val_errors['1-auc_ovr'][0])
            rec['checkpoint_metric_replay_error']=abs(replay-recorded)
            assert abs(replay-recorded)<1e-7
            pred=model.predict_proba(V)[:,1]
            measured=1-roc_auc_score(labels[iv],pred)
            rec['CPU_API_vs_CUDA_validation_random_label_AUC_difference']=abs(measured-recorded)
            model.to('cpu');payload=cloudpickle.dumps(model);del model;gc.collect();torch.cuda.empty_cache()
            loaded=cloudpickle.loads(payload);loaded.to('cuda:0')
            rec['reload_prediction_max_error']=float(np.max(np.abs(pred-loaded.predict_proba(V)[:,1])))
            assert rec['reload_prediction_max_error']<1e-7
        rec.update(status='passed_random_label_probe',completed_utc=now(),seconds=time.monotonic()-tick,
                   selected_probe_epoch=int(selection),peak_gpu_process_bytes=peak[0],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert 0<peak[0]<=CEILING
        dump(path,rec);print(json.dumps(rec,indent=2),flush=True)
    except BaseException as error:
        dump(path,dict(rec,status='failed',error=repr(error),completed_utc=now()));raise
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

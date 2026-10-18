"""Nested duration selection, then fresh outer fit with the same nine-epoch schedule."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,time,gc,threading,subprocess,resource,hashlib,signal
import numpy as np,torch,joblib,cloudpickle
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT,now,dump,sha
from research_r7_loader import STAGE,load_split
from research_r4_lean_features import LeanFeatures
from research_r8_probe import BoundedRealMLP,CAP,CEILING
from research_r10_streaming import validation_streaming_backend,ValidationStreamingModule
from pytabkit.models.training.metrics import Metrics

def main():
    plan=json.loads((ROOT/'research_r10_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    assert json.loads((ROOT/'research_r10_full_probe.json').read_text())['status']=='passed_random_label_probe'
    assert json.loads((ROOT/'research_r10_stop_probe.json').read_text())['status']=='passed'
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts'/f'r10_nested_epochs_{STAGE}';out.mkdir(exist_ok=True)
    df,raw,y,splits,reference,old_nn,fold=load_split('oof');predall=np.full(len(y),np.nan)
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
                        dump(out/'resource_failure.json',dict(ended_utc=now(),peak_gpu_process_bytes=peak[0]));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    original_train=ValidationStreamingModule.on_train_epoch_end
    original_val=ValidationStreamingModule.on_validation_epoch_end
    original_step=ValidationStreamingModule.training_step
    try:
        for k,(fi,vi) in enumerate(splits):
            run=out/f'fold{k}';run.mkdir(exist_ok=True);record=run/'result.json'
            if record.exists() and json.loads(record.read_text()).get('status')=='complete':
                rec=json.loads(record.read_text());assert rec['model_sha256']==sha(run/'model.pkl') and rec['runner_sha256']==sha(__file__)
                z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);predall[vi]=z['pred'];continue
            def timeout(signum,frame):raise TimeoutError('R10 per-outer-fold resource timeout')
            signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
            it,iv=train_test_split(fi,test_size=.1,stratify=y[fi],random_state=20261027)
            assert not np.intersect1d(it,iv).size and not np.intersect1d(np.r_[it,iv],vi).size
            np.savez_compressed(run/'inner_split.npz',train=it,valid=iv,outer_train=fi,outer_valid=vi)
            tick=time.monotonic();rec=dict(status='selecting_inner_epoch',started_utc=now(),stage=STAGE,fold=k,
                train_rows=len(fi),valid_rows=len(vi),runner_sha256=sha(__file__),
                plan_sha256=sha(ROOT/'research_r10_plan.json'),inner_split_sha256=sha(run/'inner_split.npz'),
                outer_target_used_in_selection=False,inner_validation=[],outer_epochs=[])
            dump(record,rec)
            dump(ROOT/'research_r10_state.json',dict(rec,pid=os.getpid(),active_training=True))
            print('SELECT_START',STAGE,k,flush=True)
            def step_hook(module,batch,batch_idx):
                assert all(value.device.type=='cuda' for value in batch.values())
                return original_step(module,batch,batch_idx)
            def val_hook(module):
                assert str(module.val_dl.val_x_dl.ds.device)=='cpu' and module.val_dl.val_y.device.type=='cuda'
                with torch.no_grad():
                    logits=module._postprocess_ens_pred(torch.cat(module.val_preds,dim=-2))
                    measured=float(Metrics.apply(logits[0],module.val_dl.val_y[::8][0],'1-auc_ovr'))
                original_val(module)
                rec['inner_validation'].append(dict(epoch=int(module.current_epoch)+1,
                    one_minus_auc=measured,
                    peak_gpu_process_bytes=peak[0]))
                dump(record,rec);print('INNER_EPOCH',k,rec['inner_validation'][-1],flush=True)
            ValidationStreamingModule.training_step=step_hook
            ValidationStreamingModule.on_validation_epoch_end=val_hook
            inner=LeanFeatures();X=inner.fit_transform(raw.iloc[it].reset_index(drop=True),y[it]);V=inner.transform(raw.iloc[iv].reset_index(drop=True))
            joblib.dump(inner,run/'inner_transformer.joblib')
            cfg=dict(plan['config']);cfg.pop('stop_epoch',None);cfg['val_metric_name']='1-auc_ovr'
            with validation_streaming_backend(),threadpool_limits(6):model=BoundedRealMLP(**cfg).fit(X,y[it],X_val=V,y_val=y[iv])
            selected=int(model.cv_alg_interface_.fit_params[0]['stop_epoch']['1-auc_ovr'])
            assert 1<=selected<=9 and len(rec['inner_validation'])==9
            scores=np.array([r['one_minus_auc'] for r in rec['inner_validation']])
            assert scores[selected-1]<=scores.min()+1e-7
            innerpred=model.predict_proba(V)[:,1]
            model.to('cpu')
            with open(run/'selection_model.pkl','wb') as handle:cloudpickle.dump(model,handle)
            np.savez_compressed(run/'inner_validation.npz',id=df.id.iloc[iv].to_numpy(),idx=iv,target=y[iv],pred=innerpred)
            rec.update(selected_epoch=selected,selection_model_sha256=sha(run/'selection_model.pkl'),
                       inner_transformer_sha256=sha(run/'inner_transformer.joblib'))
            del model,X,V,inner;gc.collect();torch.cuda.empty_cache()
            ValidationStreamingModule.on_validation_epoch_end=original_val
            print('SELECTED',STAGE,k,selected,flush=True)
            rec['status']='outer_refit';dump(record,rec)
            dump(ROOT/'research_r10_state.json',dict(rec,pid=os.getpid(),active_training=True))
            previous=[time.monotonic()]
            def train_hook(module):
                original_train(module);torch.cuda.synchronize()
                assert not module.creator.is_cv and module.progress.max_epochs==9
                assert str(module.train_dl.ds.device)=='cpu' and all(p.device.type=='cuda' for p in module.parameters())
                assert torch.cuda.mem_get_info()[0]>=1024**3
                moment=time.monotonic();rec['outer_epochs'].append(dict(epoch=int(module.current_epoch)+1,
                    seconds=moment-previous[0],training_parameter_devices=['cuda:0'],backing_data_device='cpu',
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_process_bytes=peak[0]));previous[0]=moment
                dump(record,rec);print('OUTER_EPOCH',k,rec['outer_epochs'][-1],flush=True)
            ValidationStreamingModule.on_train_epoch_end=train_hook
            tr=LeanFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            cfg=dict(plan['config'],stop_epoch=selected)
            with validation_streaming_backend(),threadpool_limits(6):model=BoundedRealMLP(**cfg).fit(X,y[fi])
            assert len(rec['outer_epochs'])==selected and not model.cv_alg_interface_.model.creator.is_cv
            assert np.array_equal(model.classes_,[0,1])
            pred=model.predict_proba(V)[:,1];assert np.isfinite(pred).all()
            model.to('cpu')
            with open(run/'model.pkl','wb') as handle:cloudpickle.dump(model,handle)
            del model;gc.collect();torch.cuda.empty_cache()
            with open(run/'model.pkl','rb') as handle:restored=cloudpickle.load(handle)
            restored.to('cuda:0');again=restored.predict_proba(V.iloc[:2048])[:,1]
            error=float(np.max(np.abs(pred[:2048]-again)));assert error<1e-7
            del restored;gc.collect();torch.cuda.empty_cache()
            candidate=reference[vi]+.5*(pred-old_nn[vi]);assert ((candidate>=0)&(candidate<=1)).all()
            np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],idx=vi,pred=pred,candidate=candidate)
            predall[vi]=pred
            rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-tick,
                model_sha256=sha(run/'model.pkl'),transformer_sha256=sha(run/'transformer.joblib'),
                prediction_sha256=sha(run/'validation.npz'),model_reload_error=error,
                peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            dump(record,rec);print('DONE',STAGE,k,selected,rec['elapsed_seconds'],flush=True)
            ValidationStreamingModule.on_train_epoch_end=original_train
            signal.alarm(0);del X,V,tr;gc.collect()
        assert np.isfinite(predall).all()
        candidate=reference+.5*(predall-old_nn)
        np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=candidate,reference_half=reference)
        # CUDA already initialized before CPU-only evidence helpers set visibility.
        from research_r4_common import evidence
        oldpath='r4_lean_gpu_init_oof' if STAGE=='fivefold' else 'r4_lean_gpu_init_replication3'
        official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')['candidate']
        control=evidence(y,reference,candidate,fold);direct=evidence(y,official,candidate,fold)
        result=dict(status='complete',completed_utc=now(),stage=STAGE,vs_private_half=control,
            vs_official_v7=direct,eligible=control['eligible_for_new_partition_replication'] and direct['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'))
        if STAGE=='fivefold':
            matched=np.load(ROOT/'research_artifacts/r8_stream_lean_fivefold/oof.npz');assert np.array_equal(matched['id'],df.id)
            result['matched_execution_3epoch_diagnostic']=evidence(y,matched['candidate'],candidate,fold)
        dump(out/'result.json',result);dump(ROOT/'research_r10_state.json',dict(result,active_training=False,whole_top100_task_completed=False));print(json.dumps(result,indent=2),flush=True)
    finally:
        signal.alarm(0);stop.set();watcher.join(timeout=5)
        ValidationStreamingModule.on_train_epoch_end=original_train
        ValidationStreamingModule.on_validation_epoch_end=original_val
        ValidationStreamingModule.training_step=original_step

if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump(ROOT/'research_r10_failure.json',dict(status='failed',completed_utc=now(),error=repr(error),active_training=False));raise

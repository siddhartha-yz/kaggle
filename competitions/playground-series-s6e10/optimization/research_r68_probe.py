"""Hardware-only native32, composite ranking-loss objective; no fallback."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,time,gc,threading,subprocess,resource,cloudpickle
import numpy as np,torch,pandas as pd
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT,now,dump,sha
from research_r23_loader import auxiliary
from research_r23_features import ConditionalFeatures as LeanFeatures
from research_r8_probe import BoundedRealMLP,CAP,CEILING
from research_r8_streaming import streaming_backend,StreamingTabNNModule
from research_r68_pure_loss import ranking_backend,COUNTERS,pure_ranknet
from research_r68_calibration import predict_raw,fit_bias,calibrate

def main():
    members=int(os.sys.argv[1]) if len(os.sys.argv)>1 else 32;assert members == 32
    path=ROOT/f'research_r68_probe_n{members}.json'
    rec=dict(status='running',started_utc=now(),members=members,no_quality_score=True,
             runner_sha256=sha(__file__),allocator_cap_bytes=CAP,gpu_process_ceiling_bytes=CEILING)
    dump(path,rec)
    plan=json.loads((ROOT/'research_r68_resource_plan.json').read_text())
    for name,digest in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/name)==digest,name
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event()
    from research_r23_guard import watch
    def monitor():watch(stop,peak,path)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        columns=list(pd.read_csv(ROOT/'data/train.csv',nrows=0));df=pd.read_csv(ROOT/'data/train.csv',usecols=[c for c in columns if c!='satisfaction']);raw=df.drop(columns='id');y=np.arange(len(df),dtype=np.int64)%2;aux=auxiliary(df.id);fi=np.arange(len(df));vi=fi[:4096];rec.update(official_SAT_column_loaded=False,synthetic_labels=True)
        tr=LeanFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi[:4096]].reset_index(drop=True),aux.iloc[vi[:4096]].reset_index(drop=True))
        cfg=dict(plan['config']);cfg.update(n_epochs=1,stop_epoch=1,n_ens=members)
        rec.update(config=cfg,train_rows=len(fi),features=X.shape[1],epochs=[])
        assert X.shape[1]==74
        tick=time.monotonic();original=StreamingTabNNModule.on_train_epoch_end
        def hook(module):
            original(module);torch.cuda.synchronize()
            assert all(p.device.type=='cuda' for p in module.parameters()) and str(module.train_dl.ds.device)=='cpu'
            rec['epochs'].append(dict(epoch=int(module.current_epoch)+1,seconds=time.monotonic()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_process_bytes=peak[0]))
            dump(path,rec)
        StreamingTabNNModule.on_train_epoch_end=hook
        try:
            with streaming_backend(),ranking_backend(),threadpool_limits(6):model=BoundedRealMLP(**cfg).fit(X,y[fi])
        finally:StreamingTabNNModule.on_train_epoch_end=original
        assert len(rec['epochs'])==1 and np.array_equal(model.classes_,[0,1]) and COUNTERS['calls']>0 and COUNTERS['first_logits_shape']==[32,256,2]
        rec['ranking_loss']=dict(COUNTERS,CE_coefficient=0.,pairwise_coefficient=1.)
        assert model.cv_alg_interface_.model.criterion is pure_ranknet
        train_raw=predict_raw(model,X);bias=fit_bias(train_raw,y[fi]);assert abs(calibrate(train_raw,bias).mean()-y[fi].mean())<1e-12
        rec['training_origin_calibration']=dict(rows=len(fi),external_query_chunk=2048,inner_predict_batch=1024,mean_probability_error=float(abs(calibrate(train_raw,bias).mean()-y[fi].mean())),no_official_SAT_values_parsed=True)
        pred=predict_raw(model,V);assert np.isfinite(pred).all()
        model.to('cpu');payload=cloudpickle.dumps(model);del model;gc.collect();torch.cuda.empty_cache()
        restored=cloudpickle.loads(payload);restored.to('cuda:0');replay=predict_raw(restored,V.iloc[:2048])
        error=float(np.max(np.abs(pred[:2048]-replay)));assert error<1e-7
        rec.update(status='passed_no_quality_score',completed_utc=now(),seconds=time.monotonic()-tick,
            reload_prediction_max_error=error,peak_gpu_process_bytes=peak[0],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert 0<peak[0]<=CEILING
        dump(path,rec);print(json.dumps(rec,indent=2),flush=True)
    except BaseException as error:
        rec.update(status='failed',completed_utc=now(),error=repr(error),peak_gpu_process_bytes=peak[0]);dump(path,rec);raise
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

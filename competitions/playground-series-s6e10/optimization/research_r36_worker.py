"""One immutable TabICL48 context per fresh process; no outer labels in fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import inference_config,CAP,CEILING
from research_r36_memory import streamed_ffn
from research_r34_features import features
from research_r23_loader import auxiliary
from research_r23_guard import watch
from pathlib import Path
from datetime import datetime,timezone
import torch,numpy as np,pandas as pd,json,time,gc,threading,resource,hashlib,sys
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main(stage,k,j):
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    plan=json.loads((ROOT/'research_r36_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r36_tabicl48_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y)
    fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);size=plan['context_rows'];order=np.random.default_rng(20261031+k+(0 if stage=='fivefold' else 10000)).permutation(fi);context=order[j*size:(j+1)*size]
    assert len(context)==size and np.isin(context,fi).all() and not np.isin(context,vi).any() and len(np.unique(order[:4*size]))==4*size
    run=ROOT/f'research_artifacts/r36_tabicl48_{stage}/fold{k}/context{j}';run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():
        old=json.loads((run/'result.json').read_text());assert old['plan_sha256']==sha(ROOT/'research_r36_plan.json') and old['runner_sha256']==sha(__file__)
        if old['status']=='complete':assert old['prediction_sha256']==sha(run/'validation.npz');return
    np.savez_compressed(run/'context.npz',idx=context,id=df.id.iloc[context].to_numpy());peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);watcher.start()
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        X=features(raw,auxiliary(df.id));norm=plan['norm_methods_per_context'][j];cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=[norm],random_state=42+j,offload_mode='cpu',inference_config=inference_config())
        rec=dict(status='context_fit',started_utc=now(),stage=stage,fold=k,context=j,context_rows=size,train_rows=len(fi),valid_rows=len(vi),feature_width=48,norm_method=norm,random_state=42+j,context_sha256=sha(run/'context.npz'),plan_sha256=sha(ROOT/'research_r36_plan.json'),runner_sha256=sha(__file__),outer_validation_used_for_fit_or_preprocessing=False,pretrained_weight_update=False,predicted_rows=0);dump(run/'result.json',rec)
        tick=time.monotonic()
        with threadpool_limits(6),streamed_ffn() as counters:
            m=FrozenTabICL(**cfg);m.fit(X.iloc[context],y[context]);assert np.array_equal(m.classes_,[0,1]);torch.cuda.synchronize();rec['fit_seconds']=time.monotonic()-tick
            cache_devices=sorted({str(e.key.device) for c in m.model_kv_cache_.values() for cache in [c.col_cache,c.icl_cache] if cache for e in cache.kv.values() if e.is_valid()});assert cache_devices==['cpu'];rec['cache_devices']=cache_devices
            p=np.full(len(vi),np.nan);resume_replay=None
            for start in range(0,len(vi),512):
                take=vi[start:start+512];file=run/f'query{start:06d}.npz'
                if file.exists():
                    q=np.load(file);assert np.array_equal(q['idx'],take) and np.array_equal(q['id'],df.id.iloc[take]);pv=q['pred']
                    if resume_replay is None:
                        again=m.predict_proba(X.iloc[take])[:,1];resume_replay=float(np.max(np.abs(pv-again)));assert resume_replay<1e-6
                else:
                    assert torch.cuda.mem_get_info()[0]>=1024**3
                    pv=m.predict_proba(X.iloc[take])[:,1];assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();np.savez_compressed(file,idx=take,id=df.id.iloc[take].to_numpy(),pred=pv)
                p[start:start+len(take)]=pv
                if start%(16*512)==0 or start+512>=len(vi):
                    rec.update(status='query_prediction',updated_utc=now(),predicted_rows=min(start+512,len(vi)),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0]);dump(run/'result.json',rec);dump(ROOT/'research_r36_state.json',dict(rec,pid=os.getpid(),active_work=True,active_training=False));print('PROGRESS',stage,k,j,rec['predicted_rows'],rec['seconds'],flush=True)
            again=m.predict_proba(X.iloc[vi[:512]])[:,1];repeat=float(np.max(np.abs(p[:512]-again)));assert repeat<1e-6 and 0<peak[0]<=CEILING
            np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p)
            rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,prediction_sha256=sha(run/'validation.npz'),repeat_error=repeat,resume_refit_replay_error=resume_replay,counters=dict(counters),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(run/'result.json',rec);print('CONTEXT_COMPLETE',stage,k,j,rec['seconds'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

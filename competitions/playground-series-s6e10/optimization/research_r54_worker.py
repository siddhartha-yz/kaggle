"""Built with PriorLabs-TabPFN. Frozen V2 context fit, resumable query batches; no scoring."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,time,threading,gc,resource,hashlib,sys
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from research_r48_features import covariates,ContextFeatures
from research_r51_cache_offload import streamed_cache
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main(stage,k,m):
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());assert stage in ['fivefold','threefold'] and 0<=m<4
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r53_gpu_probe.json').read_text())['status']=='passed_no_official_SAT_or_quality'
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r54_tabpfn_v2_fivefold/result.json').read_text())['eligible']
    labels=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);assert labels.satisfaction.dtype==bool;y=labels.satisfaction.to_numpy('int8');ids,frame=covariates('train');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(ids,labels.id) and np.array_equal(ids,base['id']) and np.array_equal(y,base['target'])
    fold=base['fold'];fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);shuffled=np.random.default_rng(20261054).permutation(fi);ci=shuffled[m*4096:(m+1)*4096];assert len(ci)==4096 and not np.intersect1d(ci,vi).size
    out=ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/fold{k}/context{m}';out.mkdir(parents=True,exist_ok=True);record=out/'result.json';cache=out/'validation.npz';context=out/'context.npz'
    if record.exists() and json.loads(record.read_text()).get('status')=='complete':
        r=json.loads(record.read_text());assert r['plan_sha256']==sha(ROOT/'research_r54_plan.json') and r['runner_sha256']==sha(__file__) and r['context_sha256']==sha(context) and r['transformer_sha256']==sha(out/'transformer.joblib') and r['validation_sha256']==sha(cache);return
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tr=ContextFeatures().fit(frame.iloc[ci]);X=tr.transform(frame.iloc[ci]);V=tr.transform(frame.iloc[vi]);joblib.dump(tr,out/'transformer.joblib')
    if context.exists():
        z=np.load(context);assert np.array_equal(z['idx'],ci) and np.array_equal(z['id'],ids[ci]) and np.array_equal(z['target'],y[ci]) and np.array_equal(z['X'],X,equal_nan=True)
    else:np.savez_compressed(context,idx=ci,id=ids[ci],target=y[ci],X=X,categorical_indices=np.asarray(tr.categorical_indices))
    r=dict(status='fitting_context',started_utc=now(),stage=stage,fold=k,context=m,context_rows=4096,outer_train_rows=len(fi),validation_rows=len(vi),feature_width=49,positive_class=True,parameters=plan['classifier_parameters'],plan_sha256=sha(ROOT/'research_r54_plan.json'),runner_sha256=sha(__file__),context_sha256=sha(context),transformer_sha256=sha(out/'transformer.joblib'),outer_validation_used_for_fit_or_selection=False,context_global_index_sha256=hashlib.sha256(ci.tobytes()).hexdigest(),CPU_threads=6);dump(record,r)
    done=0;p=np.full(len(vi),np.nan)
    progress=out/'progress.json';partial=out/'partial.npz'
    if progress.exists():
        old=json.loads(progress.read_text());assert old['context_sha256']==sha(context) and old['plan_sha256']==sha(ROOT/'research_r54_plan.json') and old['prediction_sha256']==sha(partial);z=np.load(partial);assert np.array_equal(z['idx'],vi);done=int(old['done_rows']);p=z['pred'];assert np.isfinite(p[:done]).all() and np.isnan(p[done:]).all()
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic();cfg=dict(plan['classifier_parameters'],model_path=str(ROOT/plan['checkpoint']),categorical_features_indices=tr.categorical_indices);assert cfg['inference_precision']=='float32';cfg['inference_precision']=torch.float32
    try:
        with streamed_cache() as counters,threadpool_limits(6):
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);model.fit(X,y[ci]);assert np.array_equal(model.classes_,[0,1]) and model.n_estimators_==1
            if done:
                resumed=model.predict_proba(V[:512])[:,1];assert np.max(np.abs(resumed-p[:512]))<1e-6
            r['status']='predicting';r['cache_build_seconds']=time.monotonic()-tick;dump(record,r)
            for lo in range(done,len(vi),512):
                hi=min(lo+512,len(vi));p[lo:hi]=model.predict_proba(V[lo:hi])[:,1];assert np.isfinite(p[lo:hi]).all() and ((p[lo:hi]>=0)&(p[lo:hi]<=1)).all()
                if hi==len(vi) or hi%8192==0:
                    temp=out/'partial.tmp.npz';np.savez_compressed(temp,idx=vi,pred=p);temp.replace(partial);pr=dict(done_rows=hi,total_rows=len(vi),updated_utc=now(),context_sha256=sha(context),plan_sha256=sha(ROOT/'research_r54_plan.json'),prediction_sha256=sha(partial));dump(progress,pr);dump(ROOT/'research_r54_state.json',dict(status='predicting',stage=stage,fold=k,context=m,pid=os.getpid(),active_training=True,peak_gpu_process_bytes=peak[0],**pr));print('QUERY',stage,k,m,hi,len(vi),flush=True)
            model.to('cpu');del model;gc.collect();torch.cuda.empty_cache();saved=np.load(context);reloaded=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);reloaded.fit(saved['X'],saved['target']);again=reloaded.predict_proba(V[:512])[:,1];error=float(np.max(np.abs(again-p[:512])));assert error<1e-6
            np.savez_compressed(cache,idx=vi,id=ids[vi],target=y[vi],pred=p);r.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,validation_sha256=sha(cache),context_reload_error=error,adapter_counters=dict(counters),peak_gpu_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert peak[0]<=CEILING and r['peak_RSS_bytes']<=8589934592;dump(record,r);print('CONTEXT_COMPLETE',stage,k,m,r['seconds'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

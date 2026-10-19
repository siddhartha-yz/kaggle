"""Only after dual qualification and32 audits: one saved context predicts test, no uploads."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,sys,time,threading,resource,numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from research_r48_features import covariates
from research_r51_cache_offload import streamed_cache
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
from research_r4_common import now,sha,dump
def main(k,m):
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());final=json.loads((ROOT/'research_r54_final_inference_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r54_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['input_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r54_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_contexts']==32 and audit['plan_sha256']==sha(ROOT/'research_r54_plan.json')
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/result.json').read_text())['eligible']
    source=ROOT/f'research_artifacts/r54_tabpfn_v2_fivefold/fold{k}/context{m}';record=json.loads((source/'result.json').read_text());assert record['context_sha256']==sha(source/'context.npz') and record['transformer_sha256']==sha(source/'transformer.joblib');ctx=np.load(source/'context.npz');tr=joblib.load(source/'transformer.joblib');ids,frame=covariates('test');assert len(ids)==299844;V=tr.transform(frame)
    out=ROOT/f'research_artifacts/r54_tabpfn_v2_full/fold{k}/context{m}';out.mkdir(parents=True,exist_ok=True);cache=out/'test.npz'
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text())
        if old['status']=='complete':assert old['context_sha256']==record['context_sha256'] and old['final_plan_sha256']==sha(ROOT/'research_r54_final_inference_plan.json') and old['test_sha256']==sha(cache);return
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=np.full(len(ids),np.nan);done=0;partial=out/'partial.npz';progress=out/'progress.json'
    if progress.exists():
        old=json.loads(progress.read_text());assert old['context_sha256']==record['context_sha256'] and old['final_plan_sha256']==sha(ROOT/'research_r54_final_inference_plan.json') and old['prediction_sha256']==sha(partial);z=np.load(partial);assert np.array_equal(z['id'],ids);p=z['pred'];done=int(old['done_rows']);assert np.isfinite(p[:done]).all() and np.isnan(p[done:]).all()
    peak=[0];stop=threading.Event();monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic();cfg=dict(plan['classifier_parameters'],model_path=str(ROOT/plan['checkpoint']),categorical_features_indices=ctx['categorical_indices'].tolist());assert cfg['inference_precision']=='float32';cfg['inference_precision']=torch.float32
    try:
        with streamed_cache() as counters,threadpool_limits(6):
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);model.fit(ctx['X'],ctx['target']);assert np.array_equal(model.classes_,[0,1]) and model.n_estimators_==1
            if done:assert np.max(np.abs(model.predict_proba(V[:512])[:,1]-p[:512]))<1e-6
            for lo in range(done,len(ids),512):
                hi=min(lo+512,len(ids));p[lo:hi]=model.predict_proba(V[lo:hi])[:,1];assert np.isfinite(p[lo:hi]).all() and ((p[lo:hi]>=0)&(p[lo:hi]<=1)).all()
                if hi==len(ids) or hi%8192==0:
                    tmp=out/'partial.tmp.npz';np.savez_compressed(tmp,id=ids,pred=p);tmp.replace(partial);dump(progress,dict(updated_utc=now(),done_rows=hi,total_rows=len(ids),context_sha256=record['context_sha256'],final_plan_sha256=sha(ROOT/'research_r54_final_inference_plan.json'),prediction_sha256=sha(partial)));print('TEST_QUERY',k,m,hi,len(ids),flush=True)
            replay=model.predict_proba(V[:512])[:,1];error=float(np.max(np.abs(p[:512]-replay)));assert error<1e-6;np.savez_compressed(cache,id=ids,pred=p);r=dict(status='complete',completed_utc=now(),fold=k,context=m,native_estimators=1,rows=len(ids),context_sha256=record['context_sha256'],transformer_sha256=record['transformer_sha256'],test_sha256=sha(cache),repeat_error=error,peak_gpu_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,adapter_counters=dict(counters),seconds=time.monotonic()-tick,final_plan_sha256=sha(ROOT/'research_r54_final_inference_plan.json'));assert peak[0]<=CEILING and r['peak_RSS_bytes']<=8589934592;dump(out/'result.json',r);print('TEST_CONTEXT_COMPLETE',k,m,r['seconds'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

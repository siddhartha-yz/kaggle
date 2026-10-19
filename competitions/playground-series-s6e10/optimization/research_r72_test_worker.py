"""Recreate one audited fivefold context and predict test with bounded, resumable512-row queries."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
from datetime import datetime,timezone
import gc,json,time,threading,hashlib,sys,resource
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import inference_config
from research_r36_memory import streamed_ffn
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_AVAILABLE,MIN_GPU_FREE
from research_r72_audit_worker import weight_digest,arr_sha
import research_r22_tabicl_offload as memory_adapter
memory_adapter.CAP=CAP
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main(k,j):
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True';plan=json.loads((ROOT/'research_r72_plan.json').read_text());final=json.loads((ROOT/'research_r72_final_inference_plan.json').read_text());audit=json.loads((ROOT/'research_r72_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_contexts']==32 and final['main_plan_sha256']==sha(ROOT/'research_r72_plan.json') and final['pre_inference_audit_sha256']==sha(ROOT/'research_r72_pre_inference_audit.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['input_sha256']}.items():assert sha(ROOT/n)==h,n
    src=ROOT/f'research_artifacts/r72_largecontext_raw21_fivefold/fold{k}/context{j}';rec=json.loads((src/'result.json').read_text());apath=ROOT/f'research_artifacts/r72_independent_audit/fivefold_fold{k}_context{j}/result.json';ar=json.loads(apath.read_text());assert rec['status']=='complete' and ar['status']=='passed' and rec['context_sha256']==ar['context_sha256']==sha(src/'context.npz') and rec['prediction_sha256']==ar['validation_sha256']==sha(src/'validation.npz');assert rec['norm_method']=='none' and rec['random_state']==42+j and rec['feature_width']==21
    out=ROOT/f'research_artifacts/r72_largecontext_test/fold{k}_context{j}';out.mkdir(parents=True,exist_ok=True);receipt=out/'result.json';prediction_plan_sha=sha(ROOT/'research_r72_final_inference_plan.json')
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['status']=='complete' and old['prediction_plan_sha256']==prediction_plan_sha and old['test_prediction_sha256']==sha(out/'test_predictions.npz');print('TEST_CACHE_REUSED',k,j,flush=True);return
    torch.set_num_threads(16);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        frame=pd.read_csv(ROOT/'data/train.csv');columns=[c for c in frame if c not in ['id','satisfaction']];assert len(columns)==21 and frame.satisfaction.dtype==bool and len(frame)==699635;ctx=np.load(src/'context.npz');saved=np.load(src/'validation.npz');context=ctx['idx'];vi=saved['idx'];assert np.array_equal(ctx['id'],frame.id.iloc[context]) and np.array_equal(saved['id'],frame.id.iloc[vi]) and np.array_equal(saved['target'],frame.satisfaction.iloc[vi].to_numpy('int8')) and not np.intersect1d(context,vi).size
        X=frame.iloc[context][columns].copy(deep=True);labels=frame.satisfaction.iloc[context].to_numpy('int8');V=frame.iloc[vi[:512]][columns].copy(deep=True);reference=saved['pred'][:512].copy();del frame,ctx,saved;gc.collect()
        cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=['none'],random_state=42+j,offload_mode='cpu',inference_config=inference_config(),n_jobs=16)
        with threadpool_limits(16),streamed_ffn() as counters:
            model=FrozenTabICL(**cfg);model.fit(X,labels);assert np.array_equal(model.classes_,[0,1]) and model.n_samples_in_==len(context) and model.n_features_in_==21 and arr_sha(model.ensemble_generator_.X_)==ar['preprocessing']['context_encoding_sha256'];del X,labels;gc.collect()
            parameter_sha=weight_digest(model.model_);assert parameter_sha==ar['weights_sha256_before_after'];replay=model.predict_proba(V)[:,1];recreation_error=float(np.max(np.abs(replay-reference)));assert recreation_error<1e-6
            test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv',usecols=['id']);assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and [c for c in test if c!='id']==columns;T=test[columns];pred=np.full(len(test),np.nan);resume_error=None;new_query_chunks=0
            for start in range(0,len(test),512):
                end=min(start+512,len(test));idx=np.arange(start,end);cache=out/f'query{start:06d}.npz'
                if cache.exists():
                    with np.load(cache) as q:assert str(q['prediction_plan_sha256'].item())==prediction_plan_sha and np.array_equal(q['idx'],idx) and np.array_equal(q['id'],test.id.iloc[idx]);p=q['pred'].copy()
                    if resume_error is None:
                        again=model.predict_proba(T.iloc[idx])[:,1];resume_error=float(np.max(np.abs(again-p)));assert resume_error<1e-6
                else:
                    assert torch.cuda.mem_get_info()[0]>=MIN_GPU_FREE;p=model.predict_proba(T.iloc[idx])[:,1];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.savez_compressed(cache,idx=idx,id=test.id.iloc[idx].to_numpy(),pred=p,prediction_plan_sha256=np.asarray(prediction_plan_sha));new_query_chunks+=1
                pred[start:end]=p
                if start%(16*512)==0 or end==len(test):dump(out/'state.json',dict(status='test_prediction',stage='fivefold',fold=k,context=j,rows=end,total=len(test),updated_utc=now(),pid=os.getpid(),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0]));print('TEST_PROGRESS',k,j,end,flush=True)
            again=model.predict_proba(T.iloc[:512])[:,1];test_repeat=float(np.max(np.abs(again-pred[:512])));valid_again=model.predict_proba(V)[:,1];valid_repeat=float(np.max(np.abs(valid_again-reference)));assert max(test_repeat,valid_repeat)<1e-6 and weight_digest(model.model_)==parameter_sha and np.isfinite(pred).all()
            np.savez_compressed(out/'test_predictions.npz',id=test.id.to_numpy(),pred=pred,prediction_plan_sha256=np.asarray(prediction_plan_sha));r=dict(status='complete',completed_utc=now(),fold=k,context=j,rows=len(test),prediction_plan_sha256=prediction_plan_sha,context_sha256=sha(src/'context.npz'),source_validation_sha256=sha(src/'validation.npz'),independent_audit_receipt_sha256=sha(apath),test_prediction_sha256=sha(out/'test_predictions.npz'),parameter_sha256=parameter_sha,context_recreation_error=recreation_error,test_A_B_A_error=test_repeat,validation_replay_after_test_error=valid_repeat,resume_query_error=resume_error,new_query_chunks=new_query_chunks,query_chunk_rows=512,new_classifier_weight_fits=0,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick);assert 0<peak[0]<=CEILING and r['peak_allocated_bytes']<=CAP and r['peak_RSS_bytes']<=8589934592;dump(receipt,r);print('TEST_CONTEXT_COMPLETE',k,j,r['seconds'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

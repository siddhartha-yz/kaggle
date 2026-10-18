"""Independent fresh-process GPU reconstruction of one frozen ICL context."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,sys,threading,resource,numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.architectures.tabpfn_v2 import TabPFNV2
from research_r48_features import covariates
from research_r51_cache_offload import streamed_cache
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
from research_r4_common import now,dump,sha
def main(stage,k,m):
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());auditplan=json.loads((ROOT/'research_r54_audit_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**auditplan['code_sha256'],**auditplan['input_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r54_static_audit.json').read_text())['status']=='passed_static_pending_GPU_replays'
    for s in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{s}/result.json').read_text())['eligible']
    run=ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/fold{k}/context{m}';r=json.loads((run/'result.json').read_text());ctx=np.load(run/'context.npz');saved=np.load(run/'validation.npz');assert r['context_sha256']==sha(run/'context.npz') and r['validation_sha256']==sha(run/'validation.npz')
    ids,frame=covariates('train');tr=joblib.load(run/'transformer.joblib');vi=saved['idx'];assert np.array_equal(ids[vi],saved['id']);start=int(np.random.default_rng(20261054+k*4+m).integers(1,len(vi)//512))*512;starts=[0,start]
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,run/'audit_resource_failure.json'),daemon=True);watcher.start();cfg=dict(plan['classifier_parameters'],model_path=str(ROOT/plan['checkpoint']),categorical_features_indices=ctx['categorical_indices'].tolist());assert cfg['inference_precision']=='float32';cfg['inference_precision']=torch.float32
    try:
        with streamed_cache() as counters,threadpool_limits(6):
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);model.fit(ctx['X'],ctx['target']);assert np.array_equal(model.classes_,[0,1]) and model.n_estimators_==1 and isinstance(model.models_[0],TabPFNV2) and model.inference_precision==torch.float32 and not model.models_[0].training
            errors=[]
            for lo in starts:
                V=tr.transform(frame.iloc[vi[lo:lo+512]]);p=model.predict_proba(V)[:,1];error=float(np.max(np.abs(p-saved['pred'][lo:lo+512])));assert error<1e-6;errors.append(dict(validation_offset=lo,rows=512,error=error))
            dtypes=sorted({str(p.dtype) for p in model.models_[0].parameters()});assert dtypes==['torch.float32'];result=dict(status='passed',audited_utc=now(),stage=stage,fold=k,context=m,model_family='TabPFN-v2',native_estimators=1,parameter_dtypes=dtypes,model_context_sha256=r['context_sha256'],validation_sha256=r['validation_sha256'],GPU_replays=errors,adapter_counters=dict(counters),positive_class='True=1',peak_gpu_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=sha(ROOT/'research_r54_plan.json'),audit_plan_sha256=sha(ROOT/'research_r54_audit_plan.json'));assert peak[0]<=CEILING and result['peak_RSS_bytes']<=8589934592;dump(run/'independent_GPU_audit.json',result);print('GPU_AUDIT_PASS',stage,k,m,errors,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

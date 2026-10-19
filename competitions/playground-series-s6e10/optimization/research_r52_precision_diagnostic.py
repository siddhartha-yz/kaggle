"""Identify cached-column batching precision effects using FP32 CPU/GPU oracles; no SAT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1' # documented speed guard; only1024 synthetic-context rows, same resource monitor
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,time,threading,gc,hashlib
from contextlib import nullcontext
import numpy as np,torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
from research_r48_features import covariates,ContextFeatures
from research_r51_cache_offload import streamed_cache
from research_r23_guard import watch
from research_r8_probe import CAP
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((ROOT/'research_r52_diagnostic_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['input_sha256']}.items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,frame=covariates('train');chosen=np.random.default_rng(20261048).permutation(len(frame));fi=chosen[:8192];vi=chosen[8192:8704];tr=ContextFeatures().fit(frame.iloc[fi]);X=tr.transform(frame.iloc[fi[:1024]]);V=tr.transform(frame.iloc[vi[:128]]);y=np.arange(1024)%2
    cfg=dict(json.loads((ROOT/'research_r50_resource_plan.json').read_text())['classifier_parameters'],model_path=str(ROOT/'research_sources/tabpfn_feasibility/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'),categorical_features_indices=tr.categorical_indices)
    values={};peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,ROOT/'research_r52_resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        for name,device,precision,adapted in [('CPU_stock','cpu',torch.float32,False),('CPU_stream','cpu',torch.float32,True),('GPU_FP32_stock','cuda:0',torch.float32,False),('GPU_FP32_stream','cuda:0',torch.float32,True),('GPU_AMP_stock','cuda:0','autocast',False),('GPU_AMP_stream','cuda:0','autocast',True)]:
            with (streamed_cache() if adapted else nullcontext()),threadpool_limits(6):
                model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**dict(cfg,device=device,inference_precision=precision));model.fit(X,y);p=model.predict_proba(V);assert p.shape==(128,2) and np.isfinite(p).all();values[name]=p;model.to('cpu');del model;gc.collect();torch.cuda.empty_cache();print('DONE',name,flush=True)
        errors={}
        for a,b in [('CPU_stock','CPU_stream'),('GPU_FP32_stock','GPU_FP32_stream'),('CPU_stock','GPU_FP32_stock'),('GPU_AMP_stock','GPU_AMP_stream'),('CPU_stock','GPU_AMP_stock'),('CPU_stock','GPU_AMP_stream')]:errors[a+'_vs_'+b]=dict(max=float(np.max(np.abs(values[a]-values[b]))),mean=float(np.mean(np.abs(values[a]-values[b]))))
        result=dict(status='diagnostic_complete_no_quality',errors=errors,peak_gpu_process_bytes=peak[0],seconds=time.monotonic()-tick,context_rows=1024,query_rows=128,features=49,official_SAT_values_parsed=False,synthetic_labels=True,quality_scores_computed=False,thresholds_not_changed=True,plan_sha256=sha(ROOT/'research_r52_diagnostic_plan.json'));(ROOT/'research_r52_precision_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

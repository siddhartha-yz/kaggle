"""Expanded-budget raw21 coherent128k-context TabICL proof; synthetic labels only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import bounded_offload,inference_config
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_AVAILABLE,MIN_GPU_FREE
import research_r22_tabicl_offload as legacy_offload
legacy_offload.CAP=CAP
from research_r36_memory import streamed_ffn
from pathlib import Path
from datetime import datetime,timezone
import torch,numpy as np,pandas as pd,json,time,gc,threading,resource,hashlib,sys
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='16'

def main(size):
    assert size==131072
    assert json.loads((ROOT/'research_r69_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_r71_probe_n32.json').read_text())['status']=='passed_no_quality_score'
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    plan=json.loads((ROOT/'research_r72_resource_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    assert size in plan['context_sizes_descending_fresh_process_only']
    out=ROOT/f'research_artifacts/r72_tabicl21_probe_{size}';out.mkdir(exist_ok=False)
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    rec=dict(status='running',pid=os.getpid(),started_utc=now(),context_rows=size,query_rows=512,no_quality_score=True,official_SAT_column_loaded=False,synthetic_labels=True,plan_sha256=sha(ROOT/'research_r72_resource_plan.json'));dump(out/'result.json',rec)
    try:
        torch.set_num_threads(16);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*1024**3
        torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cols=list(pd.read_csv(ROOT/'data/train.csv',nrows=0));df=pd.read_csv(ROOT/'data/train.csv',usecols=[c for c in cols if c!='satisfaction']);X=df.drop(columns='id');assert X.shape[1]==21;order=np.random.default_rng(20261031).permutation(len(df));context=order[:size];query=order[size:size+512];y=np.arange(size,dtype=np.int64)%2
        cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=['none'],offload_mode='cpu',inference_config=inference_config(),n_jobs=16)
        with threadpool_limits(16),streamed_ffn() as counters:
            tick=time.monotonic();m=FrozenTabICL(**cfg);m.fit(X.iloc[context],y);torch.cuda.synchronize();rec['fit_seconds']=time.monotonic()-tick
            tick=time.monotonic();p=m.predict_proba(X.iloc[query]);torch.cuda.synchronize();rec['predict512_seconds']=time.monotonic()-tick
            assert p.shape==(512,2) and np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-5
            again=m.predict_proba(X.iloc[query]);repeat=float(np.max(np.abs(p-again)));assert repeat<1e-6
            caches=sorted({str(e.key.device) for c in m.model_kv_cache_.values() for cache in [c.col_cache,c.icl_cache] if cache for e in cache.kv.values() if e.is_valid()});assert caches==['cpu']
            rec.update(CPU_cache_bytes=sum(t.numel()*t.element_size() for c in m.model_kv_cache_.values() for cache in [c.col_cache,c.icl_cache] if cache for e in cache.kv.values() for t in [e.key,e.value] if t is not None),cache_devices=caches,repeat_error=repeat,counters=dict(counters),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert 0<peak[0]<=CEILING
        rec.update(status='passed_no_quality_score',completed_utc=now(),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,estimated_fivefold_four_context_seconds=4*(5*rec['fit_seconds']+699635/512*rec['predict512_seconds']));dump(out/'result.json',rec);print(rec,flush=True)
    except torch.OutOfMemoryError as error:
        rec.update(status='allocator_OOM',completed_utc=now(),error_type=type(error).__name__,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(out/'result.json',rec);print(rec,flush=True);raise
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(int(sys.argv[1]))

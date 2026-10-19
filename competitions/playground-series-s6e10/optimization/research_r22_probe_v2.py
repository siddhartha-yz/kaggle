"""No-quality-score parity/cache isolation and larger-context resource probe."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import bounded_offload,inference_config,CAP,CEILING
import torch,numpy as np,pandas as pd,json,time,gc,threading,subprocess,resource,signal,hashlib,psutil
from threadpoolctl import threadpool_limits

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    plan=json.loads((ROOT/'research_r22_resource_plan_v2.json').read_text())
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    for n,h in plan['artifact_sha256'].items():assert sha(ROOT/n)==h
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r22_offload_probe_v2';out.mkdir(exist_ok=True)
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.2):
            query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in query.stdout.splitlines():
                try:pid,memory=[int(x.strip()) for x in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():peak[0]=max(peak[0],memory*2**20)
            available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
            rss=psutil.Process().memory_info().rss
            if peak[0]>CEILING or available<2*1024**3 or rss>8*1024**3:
                dump(out/'resource_failure.json',dict(peak_gpu_process_bytes=peak[0],available_RAM=available,RSS=rss));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
    fold=np.load(ROOT/'research_artifacts/oof_v1/folds.npz')['fold'];fi=np.flatnonzero(fold!=0)
    order=np.random.default_rng(20261030).permutation(fi);query=order[-1024:];A=order[:1024];B=order[1024:2048]
    cfg=config(ROOT,n_estimators=1,cache=True);cfg['norm_methods']=['none']
    rec=dict(status='running',pid=os.getpid(),no_quality_score=True,query_labels_used=False,allocator_cap_bytes=CAP,gpu_process_ceiling_bytes=CEILING,attempts=[])
    dump(out/'result.json',rec)
    def clear(m):
        if m is not None:del m
        gc.collect();torch.cuda.empty_cache()
    try:
        with threadpool_limits(6):
            stream_cfg=dict(cfg,offload_mode='cpu',inference_config=inference_config())
            with bounded_offload(ff_chunk_rows=128,cpu_cache=False):
                native=FrozenTabICL(**stream_cfg);native.fit(raw.iloc[A],y[A]);native_p=native.predict_proba(raw.iloc[query[:128]])
            del native;gc.collect();torch.cuda.empty_cache()
            stream_cfg=dict(cfg,offload_mode='cpu',inference_config=inference_config())
            with bounded_offload(ff_chunk_rows=128) as counters:
                m=FrozenTabICL(**stream_cfg);m.fit(raw.iloc[A],y[A]);p=m.predict_proba(raw.iloc[query[:128]])
                error=float(np.max(np.abs(p-native_p)));assert error<1e-5
                m.fit(raw.iloc[B],y[B]);m.predict_proba(raw.iloc[query[:128]]);m.fit(raw.iloc[A],y[A]);again=m.predict_proba(raw.iloc[query[:128]])
                isolation=float(np.max(np.abs(p-again)));assert isolation<1e-6
                rec['compatibility']=dict(same_schedule_native_probability_error=error,stock_auto_batch_bitwise_equivalence=False,A_B_A_error=isolation,counters=dict(counters),cache_devices=sorted({str(e.key.device) for c in m.model_kv_cache_.values() for cache in [c.col_cache,c.icl_cache] if cache for e in cache.kv.values() if e.is_valid()}))
                assert rec['compatibility']['cache_devices']==['cpu']
                del m;gc.collect();torch.cuda.empty_cache()
            dump(out/'result.json',rec)
            for size in [16384,65536]:
                assert not np.intersect1d(order[:size],query).size
                item=dict(context_rows=size,status='running');rec['attempts'].append(item);dump(out/'result.json',rec)
                tick=time.monotonic();m=None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    with bounded_offload() as counters:
                        m=FrozenTabICL(**stream_cfg);t=time.monotonic();m.fit(raw.iloc[order[:size]],y[order[:size]]);torch.cuda.synchronize();item['fit_seconds']=time.monotonic()-t
                        t=time.monotonic();p=m.predict_proba(raw.iloc[query[:512]]);torch.cuda.synchronize();item['predict512_seconds']=time.monotonic()-t
                        assert p.shape==(512,2) and np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-5
                        item['CPU_cache_bytes']=sum(t.numel()*t.element_size() for c in m.model_kv_cache_.values() for cache in [c.col_cache,c.icl_cache] if cache for e in cache.kv.values() for t in [e.key,e.value] if t is not None)
                        item['counters']=dict(counters);item['peak_allocated_bytes']=torch.cuda.max_memory_allocated();item['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    item.update(status='passed_no_quality_score',seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                except torch.OutOfMemoryError as error:item.update(status='allocator_OOM',error_type=type(error).__name__,seconds=time.monotonic()-tick)
                finally:
                    del m;gc.collect();torch.cuda.empty_cache();dump(out/'result.json',rec)
                print('ATTEMPT',item,flush=True)
            rec.update(status='complete_no_quality_score',peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(out/'result.json',rec)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump(ROOT/'research_r22_probe_v2_failure.json',dict(error=repr(error)));raise

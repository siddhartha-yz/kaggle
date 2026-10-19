"""Synthetic-only TabDPT1.3 CPU/CUDA, full training-shape and bounded inference proof."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'research_artifacts/r75_hf_offline')
import json,time,hashlib,threading,resource,gc
from datetime import datetime,timezone
import numpy as np,torch
from tabdpt import TabDPTClassifier
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_GPU_FREE
WEIGHTS=ROOT/'research_sources/tabdpt/tabdpt1_3.safetensors'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(model):
 h=hashlib.sha256()
 for n,t in sorted(model.named_parameters()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def build(device):return TabDPTClassifier(device=device,compile=False,use_flash=True,model_weight_path=str(WEIGHTS),normalizer='standard',context_reduction='subsample',verbose=False)
def main():
 assert os.environ.get('PYTORCH_ALLOC_CONF')=='expandable_segments:True';assert sha(WEIGHTS)=='97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb';assert json.loads((ROOT/'research_r74_result.json').read_text())['status']=='not_qualified'
 torch.set_num_threads(16);torch.set_num_interop_threads(1);available=torch.cuda.is_available();free=torch.cuda.mem_get_info()[0] if available else 0;host=host_available();print('PREFLIGHT',dict(CUDA=available,GPU_free_bytes=free,host_available_bytes=host),flush=True);assert available and free>=CAP+MIN_GPU_FREE and host>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;peak=[0];stop=threading.Event();thread=threading.Thread(target=watch,args=(stop,peak,ROOT/'research_r75_probe_resource_failure.json'),daemon=True);thread.start();tick=time.monotonic()
 try:
  rng=np.random.default_rng(20261075);small=rng.normal(size=(2048,21)).astype('float32');small[:,17:]=rng.integers(0,4,(2048,4));labels=np.arange(2048)%2;query=rng.normal(size=(4096,21)).astype('float32');query[:,17:]=rng.integers(0,4,(4096,4));CPU=build('cpu');before=digest(CPU.model);CPU.fit(small,labels);cp=CPU.predict_proba(query[:32],context_size=128,batch_size=32,seed=42);assert before==digest(CPU.model);del CPU;gc.collect()
  model=build('cuda');assert before==digest(model.model);model.fit(small,labels);gp=model.predict_proba(query[:32],context_size=128,batch_size=32,seed=42);cpu_error=float(np.max(np.abs(cp-gp)));assert cpu_error<.01;del small;gc.collect()
  X=rng.normal(size=(559708,21)).astype('float32');X[:,17:]=rng.integers(0,4,(len(X),4));X[::101,0]=np.nan;y=np.arange(len(X))%2;fit_start=time.monotonic();model.fit(X,y);fit_seconds=time.monotonic()-fit_start;torch.cuda.reset_peak_memory_stats();pstart=time.monotonic();pred=model.ensemble_predict_proba(query,n_ensembles=4,context_size=8192,batch_size=4096,seed=42,temperature=1.,permute_classes=True);torch.cuda.synchronize();predict_seconds=time.monotonic()-pstart;again=model.ensemble_predict_proba(query,n_ensembles=4,context_size=8192,batch_size=4096,seed=42,temperature=1.,permute_classes=True);short=model.ensemble_predict_proba(query[:512],n_ensembles=4,context_size=8192,batch_size=4096,seed=42,temperature=1.,permute_classes=True);repeat=float(np.max(np.abs(pred-again)));batch_error=float(np.max(np.abs(pred[:512]-short)));assert pred.shape==(4096,2) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all() and np.max(np.abs(pred.sum(1)-1))<1e-6;assert repeat<1e-6 and batch_error<.001 and before==digest(model.model)
  report=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),synthetic_only=True,official_SAT_loaded=False,pretrained_weight_updates=False,weights_parameters_sha256=before,CPU_GPU_small128context_probability_error=cpu_error,fullshape_training_rows=559708,feature_width=21,query_rows=4096,context_size=8192,native_ensembles=4,native_method='Official mean log-prob ensemble with class/feature permutations',compile=False,use_flash=True,fit_seconds=fit_seconds,predict4096_fourensemble_seconds=predict_seconds,estimated_full_fivefold_seconds=predict_seconds*171,repeat_error=repeat,query_batch_change_error=batch_error,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,source_sha256=sha(__file__),weights_sha256=sha(WEIGHTS),GPU=torch.cuda.get_device_name(0),limitations='GPU official BF16 inference differs from CPU FP32; no bitwise equivalence claim. Synthetic timing excludes real preprocessing and host contention. No validation score or ranking.');assert 0<peak[0]<=CEILING and report['peak_allocated_bytes']<=CAP and report['peak_RSS_bytes']<=8589934592;(ROOT/'research_r75_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
 finally:stop.set();thread.join(timeout=5)
if __name__=='__main__':main()

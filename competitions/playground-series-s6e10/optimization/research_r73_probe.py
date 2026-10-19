"""Synthetic-only causal density CUDA compatibility and resource proof; no official SAT or AUC."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import json,time,hashlib,threading,resource
from datetime import datetime,timezone
import torch
from research_r73_density_model import JointDensity
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_AVAILABLE,MIN_GPU_FREE
ROOT=Path(__file__).resolve().parent
assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*2**30
torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42);stop=threading.Event();peak=[0];thread=threading.Thread(target=watch,args=(stop,peak,ROOT/'research_r73_probe_resource_failure.json'),daemon=True);thread.start()
try:
 model=JointDensity([2,2,2,3]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);batch=4096;c=torch.stack([torch.randint(0,v+1,(batch,),device='cuda') for v in [2,2,2,3]],dim=1);n=torch.randn(batch,8,device='cuda');r=torch.randint(0,6,(batch,13),device='cuda');y=torch.randint(0,2,(batch,),device='cuda');times=[]
 for step in range(32):
  torch.cuda.synchronize();tick=time.monotonic();opt.zero_grad(set_to_none=True);loss=model.joint_loss(c,n,r,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-tick)
 model.eval()
 with torch.inference_mode():
  a=model.predict_proba(c[:512],n[:512],r[:512]);b=model.predict_proba(c[:512],n[:512],r[:512]);assert torch.isfinite(a).all();repeat=float((a-b).abs().max());assert repeat==0
 record=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),synthetic_only=True,official_SAT_loaded=False,batch_size=batch,steps=32,mean_step_seconds=sum(times[2:])/len(times[2:]),estimated_fivefold_4members_6epochs_seconds=sum(times[2:])/len(times[2:])*137*6*4*5,parameter_count=sum(p.numel() for p in model.parameters()),GPU=torch.cuda.get_device_name(0),peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,repeat_error=repeat,model_sha256=hashlib.sha256((ROOT/'research_r73_density_model.py').read_bytes()).hexdigest());assert 0<peak[0]<=CEILING and record['peak_allocated_bytes']<=CAP and record['peak_RSS_bytes']<=8589934592;(ROOT/'research_r73_probe.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
finally:stop.set();thread.join(timeout=5)

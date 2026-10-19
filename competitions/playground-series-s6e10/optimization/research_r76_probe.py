"""Synthetic-only CPU/CUDA, isolation, fine-tune resource and timing checks."""
from research_r76_model import ROOT,make_model,predict,logits,parameter_digest
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
import torch,numpy as np,time,json,threading,resource,sys,gc,os
from datetime import datetime,timezone

def main():
 torch.set_num_threads(16);torch.set_num_interop_threads(1)
 assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=6*2**30
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,ROOT/'research_artifacts/r76_probe_resource_failure.json'),daemon=True).start()
 start=time.monotonic();rng=np.random.default_rng(20261019);X=rng.normal(size=(8192,21)).astype('float32');y=(X[:,0]+X[:,1]>0).astype('int64');V=rng.normal(size=(4096,21)).astype('float32');before=None
 try:
  cpu=make_model('cpu');before=parameter_digest(cpu);p_cpu=predict(cpu,X[:128],y[:128],V[:32],device='cpu',bf16=False);del cpu;gc.collect()
  model=make_model('cuda');assert before==parameter_digest(model)
  p_gpu=predict(model,X[:128],y[:128],V[:32]);error=float(np.max(np.abs(p_cpu-p_gpu)));assert error<.02,error
  # Queries cannot influence one another or the support quantiles.
  repeat=float(np.max(np.abs(p_gpu-predict(model,X[:128],y[:128],V[:32]))))
  changed=V[:32].copy();changed[1:]*=100;isolation=float(abs(p_gpu[0]-predict(model,X[:128],y[:128],changed)[0]));batch=float(np.max(np.abs(p_gpu[:8]-predict(model,X[:128],y[:128],V[:8]))));assert max(repeat,isolation,batch)<.002
  print('COMPATIBILITY',error,repeat,isolation,batch,flush=True)
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.3,foreach=False)
  train_profiles=[]
  for ns,nq in [(512,512),(1024,512)]:
   torch.cuda.reset_peak_memory_stats();tick=time.monotonic();losses=[]
   try:
    for step in range(2):
     model.train();optimizer.zero_grad(set_to_none=True);out=logits(model,X[:ns],y[:ns],X[2048:2048+nq]);target=torch.as_tensor(y[2048:2048+nq],device='cuda');loss=torch.nn.functional.cross_entropy(out,target);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step();torch.cuda.synchronize();losses.append(float(loss.detach()))
    train_profiles.append(dict(support=ns,query=nq,status='passed',seconds_per_step=(time.monotonic()-tick)/2,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),losses=losses));print('TRAIN_PROFILE',train_profiles[-1],flush=True)
   except torch.cuda.OutOfMemoryError:
    optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();train_profiles.append(dict(support=ns,query=nq,status='allocator_OOM'));print('TRAIN_PROFILE',train_profiles[-1],flush=True);break
  assert train_profiles[0]['status']=='passed';after=parameter_digest(model);assert before!=after
  del optimizer,model,out,loss,target;gc.collect();torch.cuda.empty_cache();model=make_model('cuda');assert parameter_digest(model)==before
  inference_profiles=[]
  for ns,nq in [(1024,1024),(2048,1024),(4096,512)]:
   tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
   try:
    p=predict(model,X[:ns],y[:ns],V[:nq]);torch.cuda.synchronize();inference_profiles.append(dict(support=ns,query=nq,status='passed',seconds=time.monotonic()-tick,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved()));print('INFERENCE_PROFILE',inference_profiles[-1],flush=True)
   except torch.cuda.OutOfMemoryError:
    gc.collect();torch.cuda.empty_cache();inference_profiles.append(dict(support=ns,query=nq,status='allocator_OOM'));print('INFERENCE_PROFILE',inference_profiles[-1],flush=True);break
  assert inference_profiles[0]['status']=='passed'
  rec=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),synthetic_only=True,official_SAT_loaded=False,original_parameters_sha256=before,CPU_FP32_GPU_BF16_max_error=error,repeat_error=repeat,query_isolation_error=isolation,query_batch_error=batch,training_profiles=train_profiles,inference_profiles=inference_profiles,peak_gpu_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start,architecture='Official unmodified AutoGluon1.5 Tab2D, strict Mitra-v2 safetensors state load, torch2.9.1 SDPA, no flash-attn extension',limitations='Synthetic resource timing; reduced contexts and own fixed-step adaptation, not reproduction of official eightbag16k-context benchmark.')
  (ROOT/'research_r76_probe.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

"""Saved R68 native32 logits, CPU/GPU and default aggregation compatibility; no outer quality."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,threading,time,resource
import cloudpickle,joblib,numpy as np,pandas as pd,torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from research_r23_loader import auxiliary
from research_r70_inputs import training_frame
from research_r70_member_origin import predict_margins
from research_resource_guard_v2 import watch,host_available,MIN_GPU_FREE
import research_resource_guard_v2 as task_limits
CAP=1610612736
CEILING=2147483648
task_limits.CEILING=CEILING
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((ROOT/'research_r73_result.json').read_text())['status']=='not_qualified';torch.set_num_threads(16);torch.set_num_interop_threads(1);available=torch.cuda.is_available();free=torch.cuda.mem_get_info()[0] if available else None;host=host_available();print('PREFLIGHT',dict(cuda_available=available,GPU_free_bytes=free,required_GPU_free_bytes=CAP+MIN_GPU_FREE,host_available_bytes=host,required_host_available_bytes=6*2**30,visible=os.environ.get('CUDA_VISIBLE_DEVICES')),flush=True);assert available and free>=CAP+MIN_GPU_FREE and host>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,ROOT/'research_r74_probe_resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
 try:
  df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');aux=auxiliary(df.id);z=np.load(ROOT/'research_artifacts/r68_pure_ranknet_fivefold/oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fi=np.flatnonzero(z['fold']!=0);run=ROOT/'research_artifacts/r68_pure_ranknet_fivefold/fold0';rec=json.loads((run/'result.json').read_text())
  for n,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('calibration_training.npz','calibration_training_sha256')]:assert sha(run/n)==rec[key]
  tr=joblib.load(run/'transformer.joblib');X=training_frame(tr,raw,y,aux,fi).iloc[:2048];saved=np.load(run/'calibration_training.npz');assert np.array_equal(saved['idx'],fi) and np.array_equal(saved['id'],df.id.iloc[fi]);
  with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
  module=model.cv_alg_interface_.model;model.to('cpu');module.creator.device_info='cpu'
  with threadpool_limits(16):cpu=predict_margins(model,X.iloc[:256])
  model.to('cuda:0');module.creator.device_info='cuda:0';torch.cuda.reset_peak_memory_stats()
  with threadpool_limits(16):default=model.predict_proba(X)[:,1];native=predict_margins(model,X);again=model.predict_proba(X)[:,1]
  replay=float(np.max(np.abs(default-saved['raw_pred'][:len(X)])));mean_error=float(np.max(np.abs(expit(native).mean(0)-default)));repeat=float(np.max(np.abs(default-again)));cpu_error=float(np.max(np.abs(cpu-native[:,:256])));assert replay<2e-6 and mean_error<2e-6 and repeat<1e-7 and cpu_error<2e-5 and '_postprocess_ens_pred' not in vars(module)
  out=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),native_margin_shape=list(native.shape),CPU_GPU_native_margin_error=cpu_error,current_meanprob_replay_error=replay,native_meanprob_reconstruction_error=mean_error,default_prediction_restored_error=repeat,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,allocator_cap_bytes=CAP,GPU_process_ceiling_bytes=CEILING,minimum_GPU_free_bytes=MIN_GPU_FREE,new_classifier_fits=0,own_strict_OOF_training_queries_only=True,official_validation_AUC_computed=False,source_model_sha256=sha(run/'model.pkl'),source_sha256={n:sha(ROOT/n) for n in ['research_r74_probe.py','research_r70_member_origin.py','research_r70_inputs.py','research_resource_guard_v2.py']});assert 0<peak[0]<=CEILING and out['peak_allocated_bytes']<=CAP and out['peak_RSS_bytes']<=8589934592;(ROOT/'research_r74_probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
 finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

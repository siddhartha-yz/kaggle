"""Task-local inference runtime; never calls neural fit or modifies installed files."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import logging
for name in ['lightning.pytorch','pytorch_lightning','lightning_fabric']:logging.getLogger(name).setLevel(logging.ERROR)
import gc,json,time,threading,resource,cloudpickle,joblib,numpy as np,pandas as pd,torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from research_r92_common import ROOT,OUT,sha,dump,now,plan,refs,source,write_guard
from research_r23_loader import auxiliary
from research_r70_inputs import training_frame
from research_r70_member_origin import native_logits,predict_margins,fit_origins,predict_calibrated
from research_r68_calibration import calibrate
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from research_resume_policy_20261021 import check_training

def setup(run):
 check_training();write_guard();assert host_available()>=6*2**30
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 run.mkdir(parents=True,exist_ok=True);stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);monitor.start();return stop,peak,monitor

def inputs(stage,k,p):
 src,meta=source(stage,k,p);frame=pd.read_csv(ROOT/'data/train.csv');raw=frame.drop(columns=['id','satisfaction']);y=frame.satisfaction.to_numpy('int8');r=refs(stage)[2]
 assert np.array_equal(frame.id,r['id']) and np.array_equal(y,r['target'])
 fi=np.flatnonzero(r['fold']!=k);vi=np.flatnonzero(r['fold']==k);z=np.load(src/'calibration_training.npz')
 assert np.array_equal(z['idx'],fi) and np.array_equal(z['id'],frame.id.to_numpy()[fi]) and np.array_equal(z['target'],y[fi])
 a=auxiliary(frame.id);tr=joblib.load(src/'transformer.joblib');tick=time.monotonic()
 with threadpool_limits(6):X=training_frame(tr,raw,y,a,fi)
 reconstruct_seconds=time.monotonic()-tick;assert len(fi)==meta['train_rows'] and len(vi)==meta['valid_rows']
 return src,frame,raw,y,r,fi,vi,a,tr,X,z,reconstruct_seconds

def load_model(src,device):
 with (src/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
 module=model.cv_alg_interface_.model
 assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and module.progress.max_epochs==3 and not module.creator.is_cv
 assert not module.config.get('ens_av_before_softmax',False)
 model.to(device);module.creator.device_info=device
 return model

def margins_with_progress(model,frame,label,statefile,stage,k):
 out=np.empty((32,len(frame)),'float64');tick=time.monotonic()
 for lo in range(0,len(frame),65536):
  check_training();hi=min(lo+65536,len(frame));out[:,lo:hi]=predict_margins(model,frame.iloc[lo:hi]);write_guard()
  dump(statefile,dict(status=label,stage=stage,fold=k,pid=os.getpid(),rows_done=hi,total_rows=len(frame),updated_utc=now(),elapsed_seconds=time.monotonic()-tick))
  print('PREDICT',stage,k,label,hi,len(frame),flush=True)
 return out,time.monotonic()-tick

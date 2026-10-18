"""Window3 frozen resources, references and conditional candidate gates."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,signal
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np,pandas as pd,torch
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():return json.loads((ROOT/'research_w3_plan.json').read_text())
def start_guard():
 p=plan();assert datetime.now(timezone.utc)<datetime.fromisoformat(p['limits']['training_start_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Window3 hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(p['limits']['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
def setup_gpu():
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.version.cuda=='12.8'
 p=plan();cap=p['gpu_allocator_cap_bytes'];override=ROOT/'research_w3_resource_override.json'
 if override.exists():
  cap=json.loads(override.read_text())['gpu_allocator_cap_bytes'];assert 0<cap<=p['gpu_allocator_cap_bytes']
 free,_=torch.cuda.mem_get_info();assert free>=cap+int(1.25*1024**3),'Initial GPU headroom guard'
 torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
 return {'initial_gpu_free_bytes':free,'gpu_name':torch.cuda.get_device_name(),'cap_bytes':cap}
def gpu_epoch_guard():
 free,_=torch.cuda.mem_get_info();assert free>=1024**3,'GPU free headroom below1GiB';return free
def load_split(split):
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction'])
 if split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');r=np.load(ROOT/'research_artifacts/gpu_value_raw_oof_n2/oof.npz');assert all(np.array_equal(z[k],r[k]) for k in ['id','target','fold']);assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
  fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];ref=r['candidate'];old=r['pred']
 else:
  z=np.load(ROOT/('iteration_artifacts/split_confirm20261017.npz' if split=='confirm' else 'artifacts/split_seed42.npz'));splits=[(z['train'],z['valid'])];r=np.load(ROOT/'research_artifacts'/f'gpu_value_raw_{split}_n2/validation.npz');vi=z['valid'];assert np.array_equal(r['idx'],vi) and np.array_equal(r['id'],df.id.iloc[vi]) and np.array_equal(r['target'],y[vi]);ref=r['candidate'];old=r['pred'];fold=None
 return df,raw,y,splits,ref,old,fold
def gate(c,split):
 p=plan()['gates'];return c['delta_auc']>=p[split+'_min_delta'] and c['conditional_95pct_ci'][0]>0

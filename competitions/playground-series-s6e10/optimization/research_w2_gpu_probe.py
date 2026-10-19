"""Authorized serial limited-memory CUDA probe; no validation scores or model reuse."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,signal,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_realmlp_features import RealMLPFeatures
from research_external_ablation import external
ROOT=Path(__file__).resolve().parent
def save(x):(ROOT/'research_w2_gpu_probe_result.json').write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--allocator-gib',type=float,default=1.0);a=p.parse_args();assert 0<a.allocator_gib<=2
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.version.cuda=='12.8'
 props=torch.cuda.get_device_properties(0);cap=int(a.allocator_gib*1024**3);free,total=torch.cuda.mem_get_info();assert free>=cap+int(1.25*1024**3), 'Insufficient initial GPU headroom'
 torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def alarm(s,f):raise TimeoutError('GPU probe300second cap')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(300)
 record={'status':'preprocessing','started_utc':datetime.now(timezone.utc).isoformat(),'allocator_cap_bytes':cap,'initial_free_bytes':free,'gpu_name':props.name,'compute_capability':list(torch.cuda.get_device_capability()),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'validation_scores_used':False,'epochs':[]};save(record)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();fi=np.load(ROOT/'artifacts/split_seed42.npz')['train'];ext,_=external('both',df.id);tr=RealMLPFeatures();X=tr.fit_transform(df.drop(columns=['id','satisfaction']).iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));cfg=dict(json.loads((ROOT/'research_realmlp_w2_plan.json').read_text())['config']);cfg.update(device='cuda:0',n_epochs=3);m=RealMLP_TD_Classifier(**cfg);hook=TabNNModule.on_train_epoch_end;last=[time.monotonic()]
 def epoch_hook(module):
  hook(module);torch.cuda.synchronize();t=time.monotonic();f,_=torch.cuda.mem_get_info();record['epochs'].append({'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'free_bytes':f});last[0]=t;save(record)
  if f<1024**3:raise RuntimeError('GPU free-memory headroom below1GiB')
 TabNNModule.on_train_epoch_end=epoch_hook;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 try:
  with threadpool_limits(6):m.fit(X,y[fi]);pred=m.predict_proba(X.iloc[:1024])[:,1]
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();record.update(status='complete',fit_and_inference_seconds=time.monotonic()-start,train_rows=len(fi),feature_width=X.shape[1],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),config=cfg,score_evaluated=False,completed_utc=datetime.now(timezone.utc).isoformat());save(record);print(json.dumps(record,indent=2),flush=True)
 except BaseException as e:record.update(status='failed',error=repr(e));save(record);raise
 finally:TabNNModule.on_train_epoch_end=hook
if __name__=='__main__':main()

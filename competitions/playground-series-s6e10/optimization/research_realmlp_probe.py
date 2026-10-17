"""No-score runtime probe, training rows only; official RealMLP CPU implementation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import time,json,signal,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split
from pytabkit import RealMLP_TD_Classifier
from research_realmlp_features import RealMLPFeatures
from research_external_ablation import external
ROOT=Path(__file__).resolve().parent
signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('Probe 300s cap')));signal.alarm(300)
torch.set_num_threads(6);torch.set_num_interop_threads(1)
df=pd.read_csv('data/train.csv');y=df.satisfaction.astype('int8').to_numpy();fi=np.load('artifacts/split_seed42.npz')['train'];idx,_=train_test_split(fi,train_size=50000,stratify=y[fi],random_state=20261021);ext,_=external('both',df.id);tr=RealMLPFeatures();X=tr.fit_transform(df.drop(columns=['id','satisfaction']).iloc[idx].reset_index(drop=True),y[idx],ext.iloc[idx].reset_index(drop=True));cfg=dict(device='cpu',n_threads=6,n_cv=1,n_refit=0,n_ens=1,n_epochs=3,batch_size=1024,use_ls=False,val_fraction=0.,use_early_stopping=False,random_state=42,verbosity=2)
Path('research_realmlp_probe_plan.json').write_text(json.dumps({'training_rows':len(idx),'epochs':3,'scores_not_used':True,'config':cfg,'index_sha256':hashlib.sha256(idx.tobytes()).hexdigest()},indent=2)+'\n')
start=time.monotonic();model=RealMLP_TD_Classifier(**cfg)
with threadpool_limits(6):model.fit(X,y[idx])
elapsed=time.monotonic()-start
result={'status':'complete','rows':len(idx),'feature_width':X.shape[1],'fit_seconds':elapsed,'config':cfg,'cuda_build':torch.version.cuda,'torch_version':torch.__version__,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'fit_params':model.cv_alg_interface_.fit_params,'score_evaluated':False};Path('research_realmlp_probe_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

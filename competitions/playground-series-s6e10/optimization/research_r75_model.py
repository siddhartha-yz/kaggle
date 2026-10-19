"""Own-training raw21 encoding and explicit local/offline official TabDPT1.3 inference."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'research_artifacts/r75_hf_offline')
import numpy as np,pandas as pd,torch,hashlib
from tabdpt import TabDPTClassifier
class RawEncoder:
 def fit(self,raw):
  assert raw.shape[1]==21 and 'id' not in raw and 'satisfaction' not in raw
  self.columns=list(raw);self.cats=list(raw.select_dtypes(exclude='number'));self.nums=list(raw.select_dtypes(include='number'));assert len(self.cats)==4 and len(self.nums)==17;self.vocabulary={c:sorted(raw[c].fillna('__MISSING__').astype(str).unique().tolist()) for c in self.cats};self.fit_rows=len(raw);return self
 def transform(self,raw):
  assert list(raw)==self.columns
  out=np.empty((len(raw),21),dtype='float32')
  for j,c in enumerate(self.columns):
   if c in self.vocabulary:out[:,j]=raw[c].fillna('__MISSING__').astype(str).map({v:k for k,v in enumerate(self.vocabulary[c])}).fillna(-1).to_numpy('float32')
   else:out[:,j]=raw[c].to_numpy('float32')
  assert not np.isinf(out).any();return out
 def state(self):return dict(columns=self.columns,categorical=self.cats,numerical=self.nums,vocabulary=self.vocabulary,fit_rows=self.fit_rows,unknown_category=-1)
def make_model(device='cuda'):
 return TabDPTClassifier(device=device,compile=False,use_flash=True,model_weight_path=str(ROOT/'research_sources/tabdpt/tabdpt1_3.safetensors'),normalizer='standard',missing_indicators=False,clip_sigma=8.,feature_reduction='pca',context_reduction='subsample',verbose=False)
def predict(model,X,context=8192,batch=4096):
 p=model.ensemble_predict_proba(X,n_ensembles=4,temperature=1.,context_size=context,batch_size=batch,seed=42,permute_classes=True);assert p.shape==(len(X),2) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(1)-1))<1e-6;return p[:,1].astype('float64')
def parameter_digest(model):
 digest=hashlib.sha256()
 for name,tensor in sorted(model.named_parameters()):digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return digest.hexdigest()

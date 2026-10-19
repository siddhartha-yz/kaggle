"""Unexecuted follow-up: preserve74 inputs, encode13six-state compositions in65logratios.

Fixed Helmert contrasts use no fitted projection or SAT. No classifier is run here.
"""
import numpy as np,pandas as pd
from research_r4_lean_features import LeanFeatures
EPS=1e-6
STATE_COLUMNS=[f'rating_posterior_{i}_class{j}' for i in range(13) for j in range(6)]
COORD_COLUMNS=[f'rating_posterior_{i}_ilr{j}' for i in range(13) for j in range(5)]
H=np.zeros((5,6),dtype='float64')
for k in range(1,6):H[k-1,:k]=1/np.sqrt(k*(k+1));H[k-1,k]=-k/np.sqrt(k*(k+1))
assert np.max(np.abs(H@H.T-np.eye(5)))<1e-15 and np.max(np.abs(H.sum(1)))<1e-15

def encode_auxiliary(aux,chunk_rows=16384):
 assert aux.shape[1]==105 and list(aux.columns[-78:])==STATE_COLUMNS and aux.columns.is_unique
 result=np.empty((len(aux),92),dtype='float32');result[:,:27]=aux.iloc[:,:27].to_numpy('float32')
 for lo in range(0,len(aux),chunk_rows):
  hi=min(lo+chunk_rows,len(aux));p=aux.iloc[lo:hi,27:].to_numpy(dtype='float64',copy=True).reshape(hi-lo,13,6)
  assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.max(np.abs(p.sum(-1)-1))<1e-5
  np.maximum(p,EPS,out=p);p/=p.sum(-1,keepdims=True);np.log(p,out=p);coordinates=p@H.T;result[lo:hi,27:]=coordinates.reshape(hi-lo,65).astype('float32')
 assert np.isfinite(result).all()
 return pd.DataFrame(result,columns=list(aux.columns[:27])+COORD_COLUMNS)

def reconstruct_states(coordinates):
 z=np.asarray(coordinates,dtype='float64').reshape(-1,13,5);logp=z@H;logp-=logp.max(-1,keepdims=True);p=np.exp(logp);return p/p.sum(-1,keepdims=True)

class LogratioNeuralFeatures(LeanFeatures):
 def append(self,out,aux):
  assert aux.shape==(len(out),92) and list(aux)==self.aux_columns
  assert np.isfinite(aux.to_numpy()).all() and not set(aux)&set(out)
  extra=aux.astype('float32').copy();extra.index=out.index;result=pd.concat([out,extra],axis=1)
  assert result.shape[1]==139 and result.select_dtypes(include='number').shape[1]==117;return result
 def fit_transform(self,raw,y,aux):
  self.aux_columns=list(aux);return self.append(super().fit_transform(raw,y),aux)
 def transform(self,raw,aux):return self.append(super().transform(raw),aux)

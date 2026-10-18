"""Label-isolated synthetic training residual neighborhoods within exact Flight Distance."""
import numpy as np,pandas as pd
NAMES=['syn_kernel_residual16','syn_kernel_residual64','syn_kernel_min_distance','syn_kernel_log_pool_count']
class ResidualKernel:
 def __init__(self,raw,y,teacher,source_idx):
  assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(y)==len(teacher)==len(source_idx)
  self.columns=list(raw);self.cats=raw.select_dtypes(include=['object','string']).columns.tolist();self.nums=[c for c in raw if c not in self.cats and c!='Flight Distance'];self.numeric=self.encode(raw);self.categories=raw[self.cats].fillna('__MISSING__').astype(str).to_numpy();self.groups=raw.reset_index(drop=True).groupby('Flight Distance',sort=False).indices;self.residual=np.asarray(y,dtype='float64')-np.asarray(teacher,dtype='float64');self.prior=float(self.residual.mean());self.source_idx=np.asarray(source_idx);assert len(np.unique(self.source_idx))==len(self.source_idx)
 def encode(self,raw):
  out=[]
  for c in self.nums:
   a=raw[c].to_numpy(dtype='float64')
   if c=='Age':a=a/100.
   elif c in ['Arrival Delay in Minutes','Departure Delay in Minutes']:a=np.log1p(np.maximum(a,0))/np.log1p(360.)
   else:a=a/5.
   out.append(np.where(np.isfinite(a),a,-1.))
  return np.column_stack(out)
 def transform(self,raw,query_idx,progress=None):
  assert list(raw)==self.columns and len(raw)==len(query_idx) and not np.isin(query_idx,self.source_idx).any(),'Query overlaps label-bearing source rows'
  raw=raw.reset_index(drop=True);n=self.encode(raw);c=raw[self.cats].fillna('__MISSING__').astype(str).to_numpy();out=np.empty((len(raw),4),dtype='float64');out[:,:2]=self.prior;out[:,2]=1.;out[:,3]=0.;groups=raw.groupby('Flight Distance',sort=False).indices
  for g,(key,qi) in enumerate(groups.items()):
   oi=self.groups.get(key)
   if oi is not None:
    sn=self.numeric[oi];sc=self.categories[oi];r=self.residual[oi];k=min(64,len(oi))
    for begin in range(0,len(qi),128):
     ix=qi[begin:begin+128];d=np.zeros((len(ix),len(oi)),dtype='float64')
     for j in range(len(self.nums)):d+=np.abs(n[ix,j,None]-sn[None,:,j])
     for j in range(len(self.cats)):d+=(c[ix,j,None]!=sc[None,:,j])
     d/=len(self.nums)+len(self.cats);near=np.argsort(d,axis=1,kind='stable')[:,:k];nd=np.take_along_axis(d,near,axis=1);nr=r[near];w=np.exp(-(nd-nd[:,:1])/.05)
     for column,limit in enumerate([16,64]):
      used=min(limit,k);out[ix,column]=((w[:,:used]*nr[:,:used]).sum(axis=1)+20*self.prior)/(w[:,:used].sum(axis=1)+20)
     out[ix,2]=nd[:,0];out[ix,3]=np.log1p(len(oi))
   if progress is not None and g%200==0:progress(g,len(groups))
  assert np.isfinite(out).all();return out

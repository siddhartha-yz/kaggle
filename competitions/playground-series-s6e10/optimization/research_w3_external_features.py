"""Independent unlabeled-test Distance profiles; prediction rows remove their own contribution."""
import numpy as np,pandas as pd
from research_features import DistanceFeatures
def rowhash(raw):
 x=raw.copy()
 for c in x:x[c]=x[c].astype('float64') if pd.api.types.is_numeric_dtype(x[c]) else x[c].fillna('__MISSING__').astype(str)
 return pd.util.hash_pandas_object(x,index=False).to_numpy()
class ExternalProfiles:
 def fit(self,raw,ids):
  assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(ids) and pd.Index(ids).is_unique and raw['Flight Distance'].notna().all();self.columns=list(raw);self.n=len(raw);self.alpha=20.;self.hash_by_id=pd.Series(rowhash(raw),index=np.asarray(ids));self.numeric=[c for c in raw.select_dtypes(include='number') if c!='Flight Distance'];self.cats=raw.select_dtypes(include=['object','string']).columns.tolist();g=raw.groupby('Flight Distance',sort=False);self.counts=g.size();self.sums=g[self.numeric].sum();self.nonmissing=g[self.numeric].count();self.global_sums=raw[self.numeric].sum();self.global_n=raw[self.numeric].count();self.cat_counts={};self.cat_global={}
  for c in self.cats:
   for v in sorted(raw[c].fillna('__MISSING__').astype(str).unique()):
    a=(raw[c].fillna('__MISSING__').astype(str)==v).astype('int64');self.cat_counts[(c,v)]=a.groupby(raw['Flight Distance']).sum();self.cat_global[(c,v)]=int(a.sum())
  return self
 def transform(self,raw,source_ids=None):
  assert list(raw)==self.columns;own=source_ids is not None
  if own:assert np.array_equal(self.hash_by_id.loc[np.asarray(source_ids)].to_numpy(),rowhash(raw))
  fd=raw['Flight Distance'];n=fd.map(self.counts).fillna(0).to_numpy('float64')-(1 if own else 0);assert np.all(n>=0);out={'ext_fd_count':n,'ext_fd_logcount':np.log1p(n)}
  for j,c in enumerate(self.numeric):
   a=raw[c].to_numpy('float64');present=~np.isnan(a);v=np.where(present,a,0);s=fd.map(self.sums[c]).fillna(0).to_numpy('float64')-(v if own else 0);k=fd.map(self.nonmissing[c]).fillna(0).to_numpy('float64')-(present if own else 0);prior=(self.global_sums[c]-(v if own else 0))/(self.global_n[c]-(present if own else 0));out[f'ext_fd_mean_{j:02}']=(s+self.alpha*prior)/(k+self.alpha)
  for j,((c,v),mapping) in enumerate(self.cat_counts.items()):
   one=(raw[c].fillna('__MISSING__').astype(str)==v).to_numpy('float64');k=fd.map(mapping).fillna(0).to_numpy('float64')-(one if own else 0);prior=(self.cat_global[(c,v)]-(one if own else 0))/(self.n-(1 if own else 0));out[f'ext_fd_prop_{j:02}']=(k+self.alpha*prior)/(n+self.alpha)
  result=pd.DataFrame(out);assert np.isfinite(result.to_numpy()).all();return result.astype('float32')
class ExternalTreeFeatures:
 def __init__(self,profile):self.profile=profile
 def fit_transform(self,raw,y,ext):
  self.columns=list(raw);self.ext_cols=list(ext);self.base=DistanceFeatures('distance_stats_numeric');return self.finish(raw,self.base.fit_transform_te(raw,y),ext)
 def finish(self,raw,base,ext,source_ids=None):
  assert list(raw)==self.columns and list(ext)==self.ext_cols;out=base.copy()
  for c in ext:out[c]=ext[c].to_numpy()
  extra=self.profile.transform(raw,source_ids)
  for c in extra:out[c]=extra[c].to_numpy()
  return out
 def transform(self,raw,ext,source_ids=None):return self.finish(raw,self.base.transform_te(raw),ext,source_ids)

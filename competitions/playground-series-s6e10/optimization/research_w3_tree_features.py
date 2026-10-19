"""New tree views: numeric digits/counts or independent original target-response profiles."""
import numpy as np,pandas as pd
from research_features import DistanceFeatures
def key(s,numeric):
 return s.astype('float64').astype('string').fillna('__MISSING__') if numeric else s.fillna('__MISSING__').astype(str)
class SourceProfiles:
 def fit(self,raw,y):
  assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(y)
  self.cols=[c for c in raw if c!='Flight Distance'];self.numeric=raw.select_dtypes(include='number').columns.tolist();self.prior=float(np.mean(y));self.maps={};self.alpha=20.
  for c in self.cols:
   d=pd.DataFrame({'k':key(raw[c],c in self.numeric).to_numpy(),'y':np.asarray(y)});g=d.groupby('k',sort=False).y.agg(['sum','count']);self.maps[c]=((g['sum']+self.alpha*self.prior)/(g['count']+self.alpha),g['count'])
  return self
 def transform(self,raw):
  out={}
  for j,c in enumerate(self.cols):
   means,counts=self.maps[c];k=key(raw[c],c in self.numeric);out[f'source_p_{j:02}']=k.map(means).fillna(self.prior).to_numpy('float32');out[f'source_n_{j:02}']=np.log1p(k.map(counts).fillna(0).to_numpy()).astype('float32')
  return pd.DataFrame(out)
class TreeFeatures:
 def __init__(self,variant,profiles=None):self.variant=variant;self.profiles=profiles
 def fit_transform(self,raw,y,ext):
  assert self.variant in ['digits_counts','source_profiles'];self.columns=list(raw);self.base=DistanceFeatures('distance_stats_numeric');base=self.base.fit_transform_te(raw,y);self.ext_cols=list(ext);self.medians=raw.select_dtypes(include='number').median();self.counts={c:key(raw[c],True).value_counts(dropna=False) for c in ['Age','Departure Delay in Minutes','Arrival Delay in Minutes']};return self.finish(raw,base,ext)
 def transform(self,raw,ext):assert list(raw)==self.columns;return self.finish(raw,self.base.transform_te(raw),ext)
 def finish(self,raw,base,ext):
  assert len(raw)==len(ext) and list(ext)==self.ext_cols;out=base.copy()
  for c in ext:out[c]=ext[c].to_numpy()
  if self.variant=='digits_counts':
   fd=raw['Flight Distance'].fillna(self.medians['Flight Distance']).to_numpy('int64');age=raw.Age.fillna(self.medians['Age']).to_numpy('int64');out['w3_fd_mod10']=fd%10;out['w3_fd_mod100']=fd%100;out['w3_fd_div10']=fd//10;out['w3_age_mod10']=age%10
   for j,(c,mapping) in enumerate(self.counts.items()):out[f'w3_count_{j}']=np.log1p(key(raw[c],True).map(mapping).fillna(0).to_numpy())
  else:
   p=self.profiles.transform(raw)
   for c in p:out[c]=p[c].to_numpy()
  return out

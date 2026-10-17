"""Outer-train-only distance features and strict inner-OOF target encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RATING_CATS=['Inflight wifi service','Online boarding','Ease of Online booking','Gate location']
class DistanceFeatures:
 def __init__(self,mode='distance_category',te_columns=None,alpha=100.,inner_splits=5,inner_seed=20261019):
  self.mode=mode;self.te_columns=te_columns or [];self.alpha=alpha;self.inner_splits=inner_splits;self.inner_seed=inner_seed
 def fit(self,X):
  assert 'satisfaction' not in X and 'id' not in X
  self.columns=list(X.columns);self.cats=X.select_dtypes(include=['object','string']).columns.tolist()
  self.numeric=[c for c in X if c not in self.cats and c!='Flight Distance']
  d=X['Flight Distance'];g=X.groupby('Flight Distance',sort=False)
  self.n=len(X);self.counts=g.size();self.means=X[self.numeric].mean();self.global_props={};self.prop_maps={}
  self.group_means=(g[self.numeric].sum()+20*self.means)/(g[self.numeric].count()+20)
  for c in self.cats:
   for value in sorted(X[c].fillna('__MISSING__').astype(str).unique()):
    mask=(X[c].fillna('__MISSING__').astype(str)==value).astype(float);key=(c,value)
    self.global_props[key]=float(mask.mean());self.prop_maps[key]=(mask.groupby(d).sum()+20*mask.mean())/(self.counts+20)
  raw=self._raw(X);self.output_columns=list(raw.columns)
  self.cat_columns=raw.select_dtypes(include=['object','string']).columns.tolist()
  self.vocab={c:pd.CategoricalDtype(sorted(raw[c].fillna('__MISSING__').astype(str).unique())) for c in self.cat_columns}
  self.names={c:f'f{i:03d}' for i,c in enumerate(self.output_columns)}
  return self
 def _raw(self,X):
  out=X[self.columns].copy()
  for c in RATING_CATS:out[c+'_cat']=X[c].astype(str)
  if self.mode!='base':out['distance_category']=X['Flight Distance'].astype(str)
  if self.mode in ['distance_frequency','distance_group_stats','distance_te']:
   count=X['Flight Distance'].map(self.counts).fillna(0)
   out['distance_count']=count;out['distance_log_count']=np.log1p(count)
  if self.mode in ['distance_group_stats','distance_te']:
   for c in self.numeric:out['distance_mean_'+c]=X['Flight Distance'].map(self.group_means[c]).fillna(self.means[c])
   for (c,value),mapping in self.prop_maps.items():out['distance_prop_'+c+'_'+value]=X['Flight Distance'].map(mapping).fillna(self.global_props[(c,value)])
  return out
 def transform(self,X):
  out=self._raw(X)
  for c,dtype in self.vocab.items():out[c]=out[c].fillna('__MISSING__').astype(str).astype(dtype)
  return out.rename(columns=self.names)
 @staticmethod
 def _key(X,columns):
  if len(columns)==1:return X[columns[0]].astype(str).reset_index(drop=True)
  result=X[columns[0]].astype(str).reset_index(drop=True)
  for c in columns[1:]:result=result+'\x1f'+X[c].astype(str).reset_index(drop=True)
  return result
 def _mapping(self,keys,y):
  y=np.asarray(y,dtype=float);prior=float(y.mean())
  df=pd.DataFrame({'key':np.asarray(keys),'y':y});g=df.groupby('key',sort=False).y.agg(['sum','count'])
  return (g['sum']+self.alpha*prior)/(g['count']+self.alpha),prior
 def fit_transform_te(self,X,y,folds=None):
  self.fit(X);out=self.transform(X);self.te_maps=[];self.te_audit=[]
  y=np.asarray(y);assert len(y)==len(X)
  if self.te_columns:
   folds=list(StratifiedKFold(self.inner_splits,shuffle=True,random_state=self.inner_seed).split(np.zeros(len(y)),y)) if folds is None else list(folds)
   covered=np.zeros(len(y),dtype=int)
   for fi,vi in folds:
    assert not np.intersect1d(fi,vi).size;covered[vi]+=1
   assert np.all(covered==1)
   for j,columns in enumerate(self.te_columns):
    keys=self._key(X,columns);encoded=np.full(len(y),np.nan)
    for fi,vi in folds:
     mapping,prior=self._mapping(keys.iloc[fi],y[fi]);encoded[vi]=keys.iloc[vi].map(mapping).fillna(prior)
    assert np.isfinite(encoded).all()
    mapping,prior=self._mapping(keys,y);self.te_maps.append((columns,mapping,prior))
    out[f'te{j:02d}']=encoded
    self.te_audit.append({'columns':columns,'alpha':self.alpha,'training_encoding':'inner_oof_only','inner_splits':len(folds),'outer_mapping_rows':len(y),'inner_validation_coverage_exactly_once':True})
  return out
 def transform_te(self,X):
  out=self.transform(X)
  for j,(columns,mapping,prior) in enumerate(self.te_maps):out[f'te{j:02d}']=self._key(X,columns).map(mapping).fillna(prior).to_numpy()
  return out

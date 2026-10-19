"""Train-only normal quantiles and onehots for a bounded CPU neural comparison."""
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from research_features import DistanceFeatures
class NeuralFeatures:
 def _numeric(self,base,ext):
  x=base.select_dtypes(exclude=['category','object','string']).copy()
  for c in ext:x[c]=ext[c].to_numpy()
  return x.astype('float32')
 def _categorical(self,raw):return raw[self.catcols].fillna('__MISSING__').astype(str)
 def fit_transform(self,raw,y,ext):
  assert len(raw)==len(ext) and 'id' not in raw and 'satisfaction' not in raw
  self.distance=DistanceFeatures('distance_stats_numeric');base=self.distance.fit_transform_te(raw,y);num=self._numeric(base,ext);self.numcols=list(num)
  original_cats=raw.select_dtypes(['object','string']).columns.tolist();ratings=[c for c in raw if c not in original_cats and c not in ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']];self.catcols=original_cats+ratings
  self.imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True);filled=self.imputer.fit_transform(num)
  self.quantile=QuantileTransformer(n_quantiles=min(256,len(raw)),output_distribution='normal',subsample=min(200000,len(raw)),random_state=42);normalized=self.quantile.fit_transform(filled)
  self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32);cat=self.encoder.fit_transform(self._categorical(raw))
  x=np.concatenate([np.clip(normalized,-5,5).astype('float32'),cat],axis=1);assert np.isfinite(x).all();self.width=x.shape[1];return np.ascontiguousarray(x)
 def transform(self,raw,ext):
  assert len(raw)==len(ext);num=self._numeric(self.distance.transform_te(raw),ext);assert list(num)==self.numcols
  normalized=self.quantile.transform(self.imputer.transform(num));cat=self.encoder.transform(self._categorical(raw));x=np.concatenate([np.clip(normalized,-5,5).astype('float32'),cat],axis=1);assert x.shape[1]==self.width and np.isfinite(x).all();return np.ascontiguousarray(x)

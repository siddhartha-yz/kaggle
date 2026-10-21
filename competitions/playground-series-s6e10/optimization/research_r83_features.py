"""R83: only21 raw-value strict inner cross-fitted TE added to the frozen84."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_r21_features import AuxiliaryFeatures
class Raw21TE:
 def keys(self,raw):
  assert list(raw)==self.raw_columns
  return pd.DataFrame({f'raw{i:02}':(raw[c].astype('float64').astype('string').fillna('__MISSING__') if c in self.numeric else raw[c].fillna('__MISSING__').astype(str)).to_numpy() for i,c in enumerate(self.raw_columns)})
 def add(self,base,te):
  assert base.shape[1]==84 and te.shape==(len(base),21) and np.isfinite(te).all()
  out=base.copy()
  for j in range(21):out[f'r83_raw_te{j:02}']=te[:,j].astype('float32')
  assert out.shape[1]==105;return out
 def fit_transform(self,raw,y,ext,aux):
  assert len(raw.columns)==21 and 'id' not in raw and 'satisfaction' not in raw
  self.raw_columns=list(raw);self.numeric=raw.select_dtypes(include='number').columns.tolist();self.base=AuxiliaryFeatures()
  base=self.base.fit_transform(raw,y,ext,aux)
  self.encoder=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=20261022)
  return self.add(base,self.encoder.fit_transform(self.keys(raw),np.asarray(y)))
 def transform(self,raw,ext,aux):return self.add(self.base.transform(raw,ext,aux),self.encoder.transform(self.keys(raw)))

"""Conditional response numeric view; raw categorical twins retain original score information."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_r4_lean_features import LeanFeatures
from research_features import DistanceFeatures
from research_next_auto_te_features import RATINGS
class ResponseFeatures(LeanFeatures):
 def response_keys(self,raw):return pd.DataFrame({f'r{j:02}':DistanceFeatures._key(raw,['Class','Type of Travel','Customer Type',c]).to_numpy() for j,c in enumerate(RATINGS)})
 def fit_transform(self,raw,y):
  out=super().fit_transform(raw,y);self.response_encoder=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=20261025);p=self.response_encoder.fit_transform(self.response_keys(raw),np.asarray(y));assert p.shape==(len(raw),13) and np.isfinite(p).all()
  for j,c in enumerate(RATINGS):out[c]=p[:,j].astype('float32')
  assert out.shape[1]==47 and out.select_dtypes(include='number').shape[1]==25;return out
 def transform(self,raw):
  out=super().transform(raw);p=self.response_encoder.transform(self.response_keys(raw))
  for j,c in enumerate(RATINGS):out[c]=p[:,j].astype('float32')
  assert out.shape[1]==47 and np.isfinite(p).all();return out

"""Prepared next mechanism, NOT executed: all-column empirical-Bayes and segment-response TE."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_features import DistanceFeatures
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
class ConditionalAutoTE:
 def keys(self,raw):
  keys={}
  for i,c in enumerate(self.raw_columns):
   keys[f'raw{i:02}']=(raw[c].astype('float64').astype('string').fillna('__MISSING__') if c in self.numeric else raw[c].fillna('__MISSING__').astype(str)).to_numpy()
  out=pd.DataFrame(keys);segment=DistanceFeatures._key(raw,['Class','Type of Travel','Customer Type'])
  for j,c in enumerate(RATINGS):out[f'segment_rating{j:02}']=segment+'\x1f'+raw[c].astype('float64').astype('string').reset_index(drop=True)
  assert out.shape==(len(raw),34);return out
 def add(self,out,ext,te):
  out=out.copy();assert te.shape==(len(out),34) and np.isfinite(te).all()
  for c in ext:out[c]=ext[c].to_numpy()
  for j in range(34):out[f'auto_te{j:02}']=te[:,j].astype('float32')
  return out
 def fit_transform(self,raw,y,ext):
  assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(y);self.raw_columns=list(raw);self.numeric=raw.select_dtypes(include='number').columns.tolist();self.base=DistanceFeatures('distance_stats_numeric');base=self.base.fit_transform_te(raw,y);self.encoder=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=20261022);te=self.encoder.fit_transform(self.keys(raw),np.asarray(y));return self.add(base,ext,te)
 def transform(self,raw,ext):
  assert list(raw)==self.raw_columns;return self.add(self.base.transform_te(raw),ext,self.encoder.transform(self.keys(raw)))

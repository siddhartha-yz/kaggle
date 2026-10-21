"""R84: isolate13 prespecified segment-by-rating inner cross-fitted TE fields."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_features import DistanceFeatures
from research_r21_features import AuxiliaryFeatures
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
class Segment13TE:
 def keys(self,raw):
  assert list(raw)==self.raw_columns
  segment=DistanceFeatures._key(raw,['Class','Type of Travel','Customer Type'])
  return pd.DataFrame({f'segment_rating{j:02}':(segment+'\x1f'+raw[c].astype('float64').astype('string').reset_index(drop=True)) for j,c in enumerate(RATINGS)})
 def add(self,base,te):
  assert base.shape[1]==84 and te.shape==(len(base),13) and np.isfinite(te).all()
  out=base.copy()
  for j in range(13):out[f'r84_segment_te{j:02}']=te[:,j].astype('float32')
  assert out.shape[1]==97;return out
 def fit_transform(self,raw,y,ext,aux):
  assert len(raw.columns)==21 and 'id' not in raw and 'satisfaction' not in raw
  self.raw_columns=list(raw);self.base=AuxiliaryFeatures();base=self.base.fit_transform(raw,y,ext,aux)
  self.encoder=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=20261022)
  return self.add(base,self.encoder.fit_transform(self.keys(raw),np.asarray(y)))
 def transform(self,raw,ext,aux):return self.add(self.base.transform(raw,ext,aux),self.encoder.transform(self.keys(raw)))

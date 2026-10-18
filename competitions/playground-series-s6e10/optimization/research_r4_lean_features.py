"""New lean cross/count mechanism, all fitted strictly within outer training rows."""
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from research_features import DistanceFeatures
COMBOS=[['Class','Type of Travel','Gender'],['Class','Type of Travel','Age'],['Flight Distance','Departure/Arrival time convenient']]
class LeanFeatures:
 def _base(self,raw):
  out=raw.copy()
  for c in self.cats:out[c]=raw[c].fillna('__MISSING__').astype(str)
  for j,c in enumerate(self.numeric):
   out[c]=raw[c].fillna(0).astype('float32');out[f'value_cat_{j}']=raw[c].fillna(0).astype('int64').astype(str).to_numpy()
  for j,(c,m) in enumerate(self.counts.items()):out[f'count_{j}']=raw[c].map(m).fillna(0).to_numpy('float32')
  cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw,cols).to_numpy() for j,cols in enumerate(COMBOS)})
  out['cross_class_travel_gender']=cross.combo0.to_numpy()
  return out,cross
 def fit_transform(self,raw,y):
  assert 'satisfaction' not in raw and 'id' not in raw and len(raw)==len(y)
  self.columns=list(raw);self.cats=raw.select_dtypes(exclude='number').columns.tolist();self.numeric=raw.select_dtypes(include='number').columns.tolist();self.counts={c:raw[c].value_counts() for c in self.cats+['Leg room service']}
  out,cross=self._base(raw);self.encoder=TargetEncoder(cv=5,smooth='auto',shuffle=True,random_state=42,target_type='binary');te=self.encoder.fit_transform(cross,np.asarray(y))
  for j in range(3):out[f'cross_te_{j}']=te[:,j].astype('float32')
  self.output_columns=list(out);return out
 def transform(self,raw):
  assert list(raw)==self.columns;out,cross=self._base(raw);te=self.encoder.transform(cross)
  for j in range(3):out[f'cross_te_{j}']=te[:,j].astype('float32')
  assert list(out)==self.output_columns;return out

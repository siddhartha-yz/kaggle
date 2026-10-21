"""Current Conditional74 plusONE cached official-original SAT teacher probability."""
import numpy as np,pandas as pd
from research_r23_features import ConditionalFeatures
TEACHER='official_original_teacher_probability'
class OfficialTeacherNeuralFeatures:
 def append(self,out,aux):
  assert aux.shape==(len(out),28) and aux.columns[-1]==TEACHER
  q=aux[TEACHER].to_numpy('float32');assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
  out=out.copy();out[TEACHER]=q;assert out.shape[1]==75 and out.select_dtypes(include='number').shape[1]==53
  return out
 def fit_transform(self,raw,y,aux):
  self.aux_columns=list(aux);self.base=ConditionalFeatures();return self.append(self.base.fit_transform(raw,y,aux.iloc[:,:27]),aux)
 def transform(self,raw,aux):
  assert list(aux)==self.aux_columns
  return self.append(self.base.transform(raw,aux.iloc[:,:27]),aux)

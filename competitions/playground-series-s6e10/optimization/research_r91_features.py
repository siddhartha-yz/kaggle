"""CurrentLGB84 plusONE fixed maximum over13 cached observedrating surprisals."""
import numpy as np
from research_r21_features import AuxiliaryFeatures
SURPRISALS=[f'rating_surprisal_{j}' for j in range(13)]
FIELD='max_rating_surprisal'
class MaxSurprisal:
 def add(self,out,aux):
  assert out.shape[1]==84 and aux.shape[1]==27 and all(c in aux for c in SURPRISALS)
  values=aux[SURPRISALS].to_numpy('float32');assert np.isfinite(values).all() and (values>=0).all()
  out=out.copy();out[FIELD]=values.max(axis=1);assert out.shape[1]==85;return out
 def fit_transform(self,raw,y,ext,aux):
  self.base=AuxiliaryFeatures();out=self.add(self.base.fit_transform(raw,y,ext,aux),aux);self.columns=list(out);return out
 def transform(self,raw,ext,aux):
  out=self.add(self.base.transform(raw,ext,aux),aux);assert list(out)==self.columns;return out

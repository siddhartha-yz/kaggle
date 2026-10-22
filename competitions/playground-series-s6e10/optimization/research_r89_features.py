"""Only remove13 expected-rating mean covariates from sealed LGB84."""
from research_r21_features import AuxiliaryFeatures
REMOVED=[f'aux_rating_ev_{j}' for j in range(13)]
class DropRatingMeans:
 def remove(self,out):
  assert out.shape[1]==84 and all(c in out for c in REMOVED)
  result=out.drop(columns=REMOVED);assert result.shape[1]==71
  return result
 def fit_transform(self,raw,y,ext,aux):
  self.base=AuxiliaryFeatures();out=self.remove(self.base.fit_transform(raw,y,ext,aux));self.columns=list(out);return out
 def transform(self,raw,ext,aux):
  out=self.remove(self.base.transform(raw,ext,aux));assert list(out)==self.columns;return out

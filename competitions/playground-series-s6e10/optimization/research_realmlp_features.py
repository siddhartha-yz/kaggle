"""Outer-train Distance maps and imputation; RealMLP owns its native scaling/embeddings."""
import pandas as pd,numpy as np
from research_features import DistanceFeatures
class RealMLPFeatures:
 def fit_transform(self,raw,y,external):
  assert len(raw)==len(external) and 'id' not in raw and 'satisfaction' not in raw
  self.distance=DistanceFeatures('distance_stats_numeric');base=self.distance.fit_transform_te(raw,y);self.numeric=base.select_dtypes(include='number').columns.tolist();self.missing=[c for c in self.numeric if base[c].isna().any()];self.medians=base[self.numeric].median();self.columns=list(base);self.external_columns=list(external);return self._finish(base,external)
 def _finish(self,base,external):
  assert list(base)==self.columns and list(external)==self.external_columns and len(base)==len(external)
  base=base.copy()
  for c in self.numeric:
   if c in self.missing:base[c+'_missing']=base[c].isna().astype('float32')
   base[c]=base[c].fillna(self.medians[c]).astype('float32')
  for c in external:base[c]=external[c].to_numpy(dtype='float32')
  assert np.isfinite(base.select_dtypes(include='number').to_numpy()).all()
  return base
 def transform(self,raw,external):return self._finish(self.distance.transform_te(raw),external)

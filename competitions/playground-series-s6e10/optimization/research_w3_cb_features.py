"""Outer-training-only original numeric value categories for native ordered CatBoost."""
import numpy as np,pandas as pd
from research_features import DistanceFeatures,RATING_CATS
class CBValueFeatures:
 def fit_transform(self,raw,y,ext):
  self.columns=list(raw);self.numeric=raw.select_dtypes(include='number').columns.tolist();self.medians=raw[self.numeric].median();self.base=DistanceFeatures('distance_stats_numeric');b=self.base.fit_transform_te(raw,y);self.ext_cols=list(ext);out=self.finish(raw,b,ext);self.catcols=out.select_dtypes(include=['object','string']).columns.tolist();self.output_columns=list(out);return out
 def finish(self,raw,b,ext):
  assert list(raw)==self.columns and list(ext)==self.ext_cols;out=b.copy()
  for c in ext:out[c]=ext[c].to_numpy()
  for j,c in enumerate(self.numeric):
   if c not in RATING_CATS:out[f'value_cat_{j}']=raw[c].fillna(self.medians[c]).astype('int64').astype(str).to_numpy()
  for c in out.select_dtypes(include=['object','string','category']):out[c]=out[c].astype('object').fillna('__UNKNOWN__').astype(str)
  return out
 def transform(self,raw,ext):
  out=self.finish(raw,self.base.transform_te(raw),ext);assert list(out)==self.output_columns;return out

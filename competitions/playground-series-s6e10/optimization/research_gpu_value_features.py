"""Train-fold-only numeric-value category twins; optional audited context features."""
import numpy as np,pandas as pd
from research_realmlp_features import RealMLPFeatures
from research_features import RATING_CATS
class ValueFeatures:
 def __init__(self,variant):assert variant in ['raw','context'];self.variant=variant
 def fit_transform(self,raw,y,ext):
  assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(y)
  self.columns=list(raw);self.numeric=raw.select_dtypes(include='number').columns.tolist();self.medians=raw[self.numeric].median()
  if self.variant=='context':self.context=RealMLPFeatures();base=self.context.fit_transform(raw,y,self._external(ext))
  else:base=self._raw(raw)
  out=self._twins(raw,base);self.output_columns=list(out);return out
 def _external(self,ext):
  out=ext.copy()
  for c in ['orig_lookup0_mean','orig_lookup1_mean','orig_teacher_lgb']:
   q=np.clip(out[c].to_numpy(),1e-6,1-1e-6);out[c]=np.log(q/(1-q)).astype('float32')
  return out
 def _raw(self,raw):
  out=raw.copy()
  for c in self.numeric:out[c]=out[c].fillna(self.medians[c]).astype('float32')
  return out
 def _twins(self,raw,base):
  out=base.copy()
  for j,c in enumerate(self.numeric):
   if self.variant=='context' and c in RATING_CATS:continue
   out[f'value_cat_{j}']=raw[c].fillna(self.medians[c]).astype('int64').astype(str).to_numpy()
  assert np.isfinite(out.select_dtypes(include='number').to_numpy()).all();return out
 def transform(self,raw,ext):
  assert list(raw)==self.columns
  base=self.context.transform(raw,self._external(ext)) if self.variant=='context' else self._raw(raw)
  out=self._twins(raw,base);assert list(out)==self.output_columns;return out

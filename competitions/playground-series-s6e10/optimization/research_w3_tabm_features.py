"""Outer-training-only TabM preprocessing, independently importable for model reload."""
import numpy as np
from sklearn.preprocessing import QuantileTransformer
class TabMFeatures:
 def fit_transform(self,raw):
  assert 'id' not in raw and 'satisfaction' not in raw
  self.columns=list(raw);self.num=raw.select_dtypes(include='number').columns.tolist();self.cat=[c for c in raw if c not in self.num];self.medians=raw[self.num].median();a=raw[self.num].fillna(self.medians).to_numpy('float32');self.mask=np.any(a!=a[0],axis=0);self.qt=QuantileTransformer(n_quantiles=1000,output_distribution='normal',random_state=42,subsample=100000);self.qt.fit(a[:,self.mask]);self.vocabs={c:{v:i+1 for i,v in enumerate(sorted(raw[c].fillna('__MISSING__').astype(str).unique()))} for c in self.cat};self.cardinalities=[len(self.vocabs[c])+1 for c in self.cat];return self.transform(raw)
 def transform(self,raw):
  assert list(raw)==self.columns;a=raw[self.num].fillna(self.medians).to_numpy('float32');n=self.qt.transform(a[:,self.mask]).astype('float32');c=np.column_stack([raw[k].fillna('__MISSING__').astype(str).map(self.vocabs[k]).fillna(0).to_numpy('int64') for k in self.cat]);assert np.isfinite(n).all();return n,c

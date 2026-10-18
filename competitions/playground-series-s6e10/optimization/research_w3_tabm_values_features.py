"""Train-only numerical quantiles and original numerical-value category twins."""
import numpy as np
from research_w3_tabm_features import TabMFeatures
class TabMValueFeatures(TabMFeatures):
 def _categories(self,raw):
  out=raw.copy()
  for c in self.num:out[c]=raw[c].fillna(self.medians[c]).astype('int64').astype(str)
  for c in out:out[c]=out[c].fillna('__MISSING__').astype(str)
  return out
 def fit_transform(self,raw):
  # The base transformer fits numerical quantiles before this subclass adds twins.
  base=TabMFeatures();base.fit_transform(raw);self.__dict__.update(base.__dict__);self.cat=list(raw);cats=self._categories(raw);self.vocabs={c:{v:i+1 for i,v in enumerate(sorted(cats[c].unique()))} for c in self.cat};self.cardinalities=[len(self.vocabs[c])+1 for c in self.cat];return self.transform(raw)
 def transform(self,raw):
  assert list(raw)==self.columns;a=raw[self.num].fillna(self.medians).to_numpy('float32');n=self.qt.transform(a[:,self.mask]).astype('float32');cats=self._categories(raw);c=np.column_stack([cats[k].map(self.vocabs[k]).fillna(0).to_numpy('int64') for k in self.cat]);assert np.isfinite(n).all();return n,c

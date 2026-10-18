"""The numeric-value view with native LightGBM categories, fitted only on training."""
import pandas as pd
from research_w3_cb_features import CBValueFeatures
class LGBValueFeatures:
 def fit_transform(self,raw,y,ext):
  self.base=CBValueFeatures();x=self.base.fit_transform(raw,y,ext);self.cats=self.base.catcols;self.vocabs={c:sorted(x[c].astype(str).unique()) for c in self.cats}
  for c in self.cats:x[c]=pd.Categorical(x[c].astype(str),categories=self.vocabs[c])
  return x
 def transform(self,raw,ext):
  x=self.base.transform(raw,ext)
  for c in self.cats:x[c]=pd.Categorical(x[c].astype(str),categories=self.vocabs[c])
  return x

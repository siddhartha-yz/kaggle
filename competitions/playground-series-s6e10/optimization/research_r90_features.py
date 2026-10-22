"""AddONLYthe9missing lowcardinality rating categories to currentLGB84."""
import pandas as pd
from research_r21_features import AuxiliaryFeatures
from research_features import RATING_CATS
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
EXTRA=[c for c in RATINGS if c not in RATING_CATS]
NAMES=[f'r90_rating_cat_{j:02}' for j in range(len(EXTRA))]
assert len(EXTRA)==9
class LowCardinalRatingCategories:
 def fit_vocab(self,raw):
  self.vocabs={c:pd.CategoricalDtype(sorted(raw[c].fillna('__MISSING__').astype(str).unique())) for c in EXTRA}
 def add(self,out,raw):
  assert out.shape[1]==84
  out=out.copy()
  for c,name in zip(EXTRA,NAMES):out[name]=pd.Categorical(raw[c].fillna('__MISSING__').astype(str),dtype=self.vocabs[c])
  assert out.shape[1]==93;return out
 def fit_transform(self,raw,y,ext,aux):
  self.fit_vocab(raw);self.base=AuxiliaryFeatures();out=self.add(self.base.fit_transform(raw,y,ext,aux),raw);self.columns=list(out);return out
 def transform(self,raw,ext,aux):
  out=self.add(self.base.transform(raw,ext,aux),raw);assert list(out)==self.columns;return out

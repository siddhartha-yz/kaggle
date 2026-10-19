"""Train-fold-only rating distributions within Flight Distance groups."""
import pandas as pd
from research_features import DistanceFeatures
class DistanceDistributions(DistanceFeatures):
 def __init__(self):super().__init__('distance_stats_numeric')
 def fit(self,X):
  cats=X.select_dtypes(['object','string']).columns.tolist();ratings=[c for c in X if c not in cats and c not in ['Age','Flight Distance','Arrival Delay in Minutes','Departure Delay in Minutes']]
  self.rating_prop_maps={};self.rating_prop_prior={};d=X['Flight Distance'];counts=d.value_counts()
  for c in ratings:
   assert X[c].notna().all() and X[c].isin(range(6)).all(), 'Rating domain must be 0..5'
   for value in range(6):
    indicator=(X[c]==value).astype(float);prior=float(indicator.mean());key=(c,value);self.rating_prop_prior[key]=prior;self.rating_prop_maps[key]=(indicator.groupby(d).sum()+20*prior)/(counts+20)
  return super().fit(X)
 def _raw(self,X):
  out=super()._raw(X)
  extra={f'distance_rating_prop_{c}_{value}':X['Flight Distance'].map(mapping).fillna(self.rating_prop_prior[(c,value)]) for (c,value),mapping in self.rating_prop_maps.items()}
  return pd.concat([out,pd.DataFrame(extra,index=X.index)],axis=1)

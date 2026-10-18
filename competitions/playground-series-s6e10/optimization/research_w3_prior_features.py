"""Fold-fitted raw-value view plus normalized independent teacher logit and functional offset."""
import numpy as np
from research_w3_tabm_values_features import TabMValueFeatures
class PriorFeatures:
 @staticmethod
 def logit(p):
  p=np.clip(np.asarray(p,dtype='float64'),1e-6,1-1e-6);return np.log(p/(1-p))
 def fit_transform(self,raw,teacher):
  assert len(raw)==len(teacher);self.base=TabMValueFeatures();n,c=self.base.fit_transform(raw);p=self.logit(teacher);self.mean=float(p.mean());self.std=max(float(p.std()),1e-3);self.cardinalities=self.base.cardinalities;return self.finish(n,c,p)
 def finish(self,n,c,p):
  extra=np.clip((p-self.mean)/self.std,-6,6).astype('float32');return np.column_stack([n,extra]).astype('float32'),c,p.astype('float32')
 def transform(self,raw,teacher):
  assert len(raw)==len(teacher);n,c=self.base.transform(raw);return self.finish(n,c,self.logit(teacher))

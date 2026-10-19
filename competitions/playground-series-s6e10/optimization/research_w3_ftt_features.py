"""Outer-training-only raw value-token features and independent source-teacher token."""
import numpy as np
from research_w3_tabm_values_features import TabMValueFeatures
class FTTFeatures:
 def fit_transform(self,raw,teacher):
  self.base=TabMValueFeatures();x=self.base.fit_transform(raw);self.cardinalities=self.base.cardinalities;self.mu=float(teacher.mean());self.sd=max(float(teacher.std()),1e-6);return self.join(x,teacher)
 def join(self,x,teacher):return np.column_stack([x[0],((teacher-self.mu)/self.sd).astype('float32')]).astype('float32'),x[1]
 def transform(self,raw,teacher):return self.join(self.base.transform(raw),teacher)

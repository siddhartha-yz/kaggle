"""Repair ONLY ArrivalDelay value-category NaN/zero collision in validated74."""
from research_r23_features import ConditionalFeatures
ARRIVAL='Arrival Delay in Minutes'
MISSING='__MISSING__'
class MissingCategoryNeuralFeatures(ConditionalFeatures):
 def repair(self,out,raw):
  self.repaired_column=f'value_cat_{self.numeric.index(ARRIVAL)}'
  assert out.shape[1]==74 and self.repaired_column in out
  out=out.copy();out.loc[raw[ARRIVAL].isna().to_numpy(),self.repaired_column]=MISSING
  return out
 def fit_transform(self,raw,y,aux):return self.repair(super().fit_transform(raw,y,aux),raw)
 def transform(self,raw,aux):return self.repair(super().transform(raw,aux),raw)

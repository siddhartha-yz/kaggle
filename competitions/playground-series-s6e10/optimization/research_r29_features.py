"""Lean47+validatedconditional27+six source-only missing-service probabilities."""
import numpy as np,pandas as pd
from research_r4_lean_features import LeanFeatures
class MissingServiceFeatures(LeanFeatures):
    def append(self,out,aux):
        assert aux.shape==(len(out),33) and list(aux)==self.aux_columns and np.isfinite(aux.to_numpy()).all()
        assert not set(aux)&set(out)
        extra=aux.astype('float32').copy();extra.index=out.index;out=pd.concat([out,extra],axis=1)
        assert out.shape[1]==80 and out.select_dtypes(include='number').shape[1]==58
        return out
    def fit_transform(self,raw,y,aux):
        self.aux_columns=list(aux);return self.append(super().fit_transform(raw,y),aux)
    def transform(self,raw,aux):return self.append(super().transform(raw),aux)

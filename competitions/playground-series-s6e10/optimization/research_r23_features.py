"""Lean47 plus all27 target-free expected ratings and observed-rating surprisals."""
import numpy as np
from research_r4_lean_features import LeanFeatures
class ConditionalFeatures(LeanFeatures):
    def append(self,out,aux):
        assert aux.shape==(len(out),27) and list(aux)==self.aux_columns
        assert np.isfinite(aux.to_numpy()).all()
        for c in self.aux_columns:
            assert c not in out
            out[c]=aux[c].to_numpy('float32')
        assert out.shape[1]==74 and out.select_dtypes(include='number').shape[1]==52
        return out
    def fit_transform(self,raw,y,aux):
        self.aux_columns=list(aux)
        return self.append(super().fit_transform(raw,y),aux)
    def transform(self,raw,aux):return self.append(super().transform(raw),aux)

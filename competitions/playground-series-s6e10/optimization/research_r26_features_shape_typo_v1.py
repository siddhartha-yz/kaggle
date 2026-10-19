"""Lean47 plus current27 conditional fields plus78 class probabilities."""
import numpy as np,pandas as pd
from research_r4_lean_features import LeanFeatures
class PosteriorNeuralFeatures(LeanFeatures):
    def append(self,out,aux):
        assert aux.shape==(len(out),105) and list(aux)==self.aux_columns
        assert np.isfinite(aux.to_numpy()).all()
        assert not set(self.aux_columns)&set(out.columns)
        extra=aux.astype('float32').copy();extra.index=out.index
        out=pd.concat([out,extra],axis=1)
        assert out.shape[1]==1130 and out.select_dtypes(include='number').shape[1]==130
        return out
    def fit_transform(self,raw,y,aux):
        self.aux_columns=list(aux)
        return self.append(super().fit_transform(raw,y),aux)
    def transform(self,raw,aux):return self.append(super().transform(raw),aux)

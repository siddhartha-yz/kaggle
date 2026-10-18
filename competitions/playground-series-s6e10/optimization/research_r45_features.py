"""Lean47+validatedconditional27+ONEboundednewpublicsourceSATprobability."""
import numpy as np,pandas as pd
from research_r4_lean_features import LeanFeatures
class SourceNeuralFeatures(LeanFeatures):
    def append(self,out,aux):
        assert aux.shape==(len(out),28) and list(aux)==self.aux_columns and np.isfinite(aux.to_numpy()).all() and not set(aux)&set(out)
        extra=aux.astype('float32').copy();extra.index=out.index;out=pd.concat([out,extra],axis=1)
        assert out.shape[1]==75 and out.select_dtypes(include='number').shape[1]==53;return out
    def fit_transform(self,raw,y,aux):
        self.aux_columns=list(aux);return self.append(super().fit_transform(raw,y),aux)
    def transform(self,raw,aux):return self.append(super().transform(raw),aux)

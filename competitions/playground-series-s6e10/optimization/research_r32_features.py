"""Validated neural74 fields followed by own-training TabM quantiles/vocabularies."""
from research_r23_features import ConditionalFeatures
from research_w3_tabm_features import TabMFeatures
class ConditionalTabMFeatures:
    def fit_transform(self,raw,y,aux):
        self.lean=ConditionalFeatures();expanded=self.lean.fit_transform(raw,y,aux)
        self.tabm=TabMFeatures();out=self.tabm.fit_transform(expanded)
        self.cardinalities=self.tabm.cardinalities
        assert expanded.shape[1]==74 and len(self.cardinalities)==22
        assert out[0].shape[1]<=52 and out[1].shape[1]==22
        return out
    def transform(self,raw,aux):
        return self.tabm.transform(self.lean.transform(raw,aux))

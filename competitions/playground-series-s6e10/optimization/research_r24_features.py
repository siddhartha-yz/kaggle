"""Existing main57 + conditionalmeans13 +selfsurprisal14 + uncertainty28 =112."""
from research_features import DistanceFeatures
class UncertaintyFeatures:
    def finish(self,base,ext,aux):
        out=base.copy()
        for c in ext:out[c]=ext[c].to_numpy()
        for c in aux:out[c]=aux[c].to_numpy('float32')
        assert out.shape[1]==112
        return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=DistanceFeatures('distance_stats_numeric')
        out=self.finish(self.base.fit_transform_te(raw,y),ext,aux);self.columns=list(out);return out
    def transform(self,raw,ext,aux):
        out=self.finish(self.base.transform_te(raw),ext,aux);assert list(out)==self.columns;return out

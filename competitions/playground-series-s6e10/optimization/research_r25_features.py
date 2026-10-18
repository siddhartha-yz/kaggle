"""Existing main84 plus all6 class probabilities for13 ratings =162."""
from research_features import DistanceFeatures
class PosteriorFeatures:
    def finish(self,base,ext,aux):
        out=base.copy()
        for c in ext:out[c]=ext[c].to_numpy()
        for c in aux:out[c]=aux[c].to_numpy('float32')
        assert out.shape[1]==162
        return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=DistanceFeatures('distance_stats_numeric')
        out=self.finish(self.base.fit_transform_te(raw,y),ext,aux);self.columns=list(out);return out
    def transform(self,raw,ext,aux):
        out=self.finish(self.base.transform_te(raw),ext,aux);assert list(out)==self.columns;return out

"""Unchangedvalidated84 fields plus13 signedconditional positive-rating ranks."""
from research_r21_features import AuxiliaryFeatures
class RankFeatures:
    def append(self,out,aux):
        assert out.shape[1]==84 and aux.shape==(len(out),40)
        for c in aux.columns[27:]:out[c]=aux[c].to_numpy('float32')
        assert out.shape[1]==97;return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=AuxiliaryFeatures();out=self.base.fit_transform(raw,y,ext,aux.iloc[:,:27]);return self.append(out,aux)
    def transform(self,raw,ext,aux):return self.append(self.base.transform(raw,ext,aux.iloc[:,:27]),aux)

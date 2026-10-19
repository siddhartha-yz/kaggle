"""Append all13 independent-source class-conditional likelihood ratios to certified84."""
from research_r21_features import AuxiliaryFeatures
class ClassLikelihoodFeatures:
    def finish(self,base,extra):
        assert base.shape[1]==84 and extra.shape==(len(base),13)
        for c in extra:assert c not in base;base[c]=extra[c].to_numpy('float32')
        assert base.shape[1]==97
        return base
    def fit_transform(self,raw,y,ext,aux):
        assert aux.shape[1]==40;self.base=AuxiliaryFeatures();out=self.finish(self.base.fit_transform(raw,y,ext,aux.iloc[:,:27]),aux.iloc[:,27:]);self.columns=list(out);return out
    def transform(self,raw,ext,aux):
        out=self.finish(self.base.transform(raw,ext,aux.iloc[:,:27]),aux.iloc[:,27:]);assert list(out)==self.columns;return out

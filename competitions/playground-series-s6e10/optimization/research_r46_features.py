"""Exact current LGB84 plus 27 nonrating-cohort-conditioned rating summaries."""
from research_r21_features import AuxiliaryFeatures
class CohortFeatures:
    def finish(self,out,aux):
        assert out.shape[1]==84 and aux.shape==(len(out),54)
        for c in aux.columns[27:]:
            assert c.startswith('cohort_rating_') and c not in out
            out[c]=aux[c].to_numpy('float32')
        assert out.shape[1]==111;return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=AuxiliaryFeatures();return self.finish(self.base.fit_transform(raw,y,ext,aux.iloc[:,:27]),aux)
    def transform(self,raw,ext,aux):return self.finish(self.base.transform(raw,ext,aux.iloc[:,:27]),aux)

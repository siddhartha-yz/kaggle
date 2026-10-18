"""Unchanged84plusoneindependentpublicsourceSATteacherfeature."""
from research_r21_features import AuxiliaryFeatures
class NewSourceFeatures:
    def finish(self,out,aux):
        assert out.shape[1]==84 and aux.shape==(len(out),28)
        out['invistico_teacher_lgb']=aux.iloc[:,-1].to_numpy('float64');assert out.shape[1]==85;return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=AuxiliaryFeatures();return self.finish(self.base.fit_transform(raw,y,ext,aux.iloc[:,:27]),aux)
    def transform(self,raw,ext,aux):return self.finish(self.base.transform(raw,ext,aux.iloc[:,:27]),aux)

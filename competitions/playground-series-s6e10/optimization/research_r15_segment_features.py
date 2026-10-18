"""One cross-fitted customer-segment prior; existing Lean47 fields remain intact."""
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from research_features import DistanceFeatures
from research_r4_lean_features import LeanFeatures
SEGMENT=['Class','Type of Travel','Customer Type']

class SegmentPrior:
    def fit(self,raw,y):
        assert 'id' not in raw and 'satisfaction' not in raw and len(raw)==len(y)
        y=np.asarray(y,dtype='float64');assert np.isin(y,[0,1]).all()
        self.target_mean_=float(y.mean());self.n_train=len(y)
        keys=DistanceFeatures._key(raw,SEGMENT)
        self.table=pd.DataFrame({'key':keys.to_numpy(),'y':y}).groupby('key').y.agg(['sum','count'])
        self.mapping=(self.table['sum']+20*self.target_mean_)/(self.table['count']+20)
        return self
    def transform(self,raw):
        return DistanceFeatures._key(raw,SEGMENT).map(self.mapping).fillna(self.target_mean_).to_numpy('float32')

def crossfit(raw,y):
    out=np.full(len(raw),np.nan,'float32');fold=np.full(len(raw),-1,'int8')
    for k,(it,iv) in enumerate(KFold(5,shuffle=True,random_state=42).split(raw)):
        assert not np.intersect1d(it,iv).size
        estimator=SegmentPrior().fit(raw.iloc[it].reset_index(drop=True),np.asarray(y)[it])
        out[iv]=estimator.transform(raw.iloc[iv].reset_index(drop=True));fold[iv]=k
    assert np.isfinite(out).all() and (fold>=0).all()
    return out,fold

class SegmentFeatures(LeanFeatures):
    def fit_transform(self,raw,y):
        out=super().fit_transform(raw,y)
        values,self.segment_crossfit_fold=crossfit(raw,y)
        self.customer_prior=SegmentPrior().fit(raw,y)
        out['customer_segment_prior']=values
        assert out.shape[1]==48 and out.select_dtypes(include='number').shape[1]==26
        return out
    def transform(self,raw):
        out=super().transform(raw)
        out['customer_segment_prior']=self.customer_prior.transform(raw)
        assert out.shape[1]==48
        return out

"""Log-precondition only two zero-IQR delay numerics; retain every raw category identity."""
import numpy as np
from research_r23_features import ConditionalFeatures
DELAYS=['Departure Delay in Minutes','Arrival Delay in Minutes']
class DelayLogFeatures(ConditionalFeatures):
    def precondition(self,out):
        for c in DELAYS:
            value=out[c].to_numpy();assert np.isfinite(value).all() and (value>=0).all()
            identity=f'value_cat_{self.numeric.index(c)}';assert np.array_equal(out[identity].astype('int64').to_numpy(),value.astype('int64'))
            out[c]=np.log1p(value.astype('float64')).astype('float32')
        assert out.shape[1]==74 and out.select_dtypes(include='number').shape[1]==52
        return out
    def fit_transform(self,raw,y,aux):return self.precondition(super().fit_transform(raw,y,aux))
    def transform(self,raw,aux):return self.precondition(super().transform(raw,aux))

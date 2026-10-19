"""Lean47 plus four raw-numeric frequency fields; train queries leave themselves out."""
import numpy as np
from research_r4_lean_features import LeanFeatures
COUNT_COLUMNS=['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']

class NumericCountFeatures(LeanFeatures):
    def add(self,out,raw,training):
        for j,c in enumerate(COUNT_COLUMNS):
            count=raw[c].map(self.frequency_maps[c]).fillna(0).to_numpy('float32')
            if training:count-=1
            assert np.isfinite(count).all() and (count>=0).all()
            out[f'numeric_frequency_{j}']=count
        assert out.shape[1]==51 and out.select_dtypes(include='number').shape[1]==29
        return out

    def fit_transform(self,raw,y):
        self.frequency_maps={c:raw[c].value_counts(dropna=False) for c in COUNT_COLUMNS}
        return self.add(super().fit_transform(raw,y),raw,True)

    def transform(self,raw):
        return self.add(super().transform(raw),raw,False)

"""Pool outer-training and official-test covariates for the SAME distance fields.

Keep baseline smoothing20 and outer-training category vocabulary. Every pool
member receives leave-query-out statistics; outer validation is never pooled.
No target enters these fits. Previous W3 test-only extra fields were different.
"""
import numpy as np
import pandas as pd
from research_features import DistanceFeatures

class PopulationDistance(DistanceFeatures):
    def __init__(self):super().__init__('distance_stats_numeric')

    def fit_population(self,raw,reference):
        assert list(raw)==list(reference) and 'id' not in raw and 'satisfaction' not in raw
        super().fit(raw)
        pool=pd.concat([raw,reference],ignore_index=True)
        self.pool_n=len(pool);self.reference_rows=len(reference);distance=pool['Flight Distance']
        group=pool.groupby('Flight Distance',sort=False)
        self.pool_counts=group.size()
        self.pool_sums=group[self.numeric].sum();self.pool_valid=group[self.numeric].count()
        self.global_sums=pool[self.numeric].sum();self.global_valid=pool[self.numeric].count()
        self.pool_props={};self.pool_global_props={}
        for key in self.prop_maps:
            c,value=key
            flag=(pool[c].fillna('__MISSING__').astype(str)==value).astype('int64')
            self.pool_props[key]=flag.groupby(distance).sum()
            self.pool_global_props[key]=int(flag.sum())
        assert self.pool_counts.sum()==self.pool_n
        return self

    def transform_population(self,raw,included=False):
        assert list(raw)==self.columns
        out=self._raw(raw)
        distance=raw['Flight Distance'];subtract=1 if included else 0
        count=distance.map(self.pool_counts).fillna(0).to_numpy('float64')-subtract
        assert (count>=0).all()
        global_n=self.pool_n-subtract
        out['distance_count']=count
        out['distance_log_count']=np.log1p(count)
        for c in self.numeric:
            value=raw[c].to_numpy('float64');valid=np.isfinite(value)
            remove=np.where(valid,value,0) if included else np.zeros(len(raw))
            remove_valid=valid.astype('int64') if included else np.zeros(len(raw),'int64')
            sums=distance.map(self.pool_sums[c]).fillna(0).to_numpy()-remove
            counts=distance.map(self.pool_valid[c]).fillna(0).to_numpy()-remove_valid
            prior=(self.global_sums[c]-remove)/(self.global_valid[c]-remove_valid)
            assert (counts>=0).all()
            out['distance_mean_'+c]=(sums+20*prior)/(counts+20)
        for key,mapping in self.pool_props.items():
            c,value=key
            remove=(raw[c].fillna('__MISSING__').astype(str)==value).to_numpy('int64') if included else np.zeros(len(raw),'int64')
            group_count=distance.map(mapping).fillna(0).to_numpy()-remove
            prior=(self.pool_global_props[key]-remove)/global_n
            out['distance_prop_'+c+'_'+value]=(group_count+20*prior)/(count+20)
        for c,dtype in self.vocab.items():out[c]=out[c].fillna('__MISSING__').astype(str).astype(dtype)
        assert list(out)==self.output_columns
        return out.rename(columns=self.names)

class PopulationFeatures:
    def __init__(self,reference):
        assert 'id' not in reference and 'satisfaction' not in reference
        self.reference=reference

    @staticmethod
    def finish(out,ext):
        out=out.copy()
        for c in ext:out[c]=ext[c].to_numpy()
        return out

    def fit_transform(self,raw,y,ext):
        assert len(raw)==len(y)
        self.base=PopulationDistance().fit_population(raw,self.reference)
        self.reference=None
        return self.finish(self.base.transform_population(raw,included=True),ext)

    def transform(self,raw,ext,included=False):
        return self.finish(self.base.transform_population(raw,included=included),ext)

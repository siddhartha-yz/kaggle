"""Existing57 CatBoost fields, with optional single nominal distance twin."""
from research_features import DistanceFeatures
class CatDistanceFeatures:
    def __init__(self,nominal):self.nominal=nominal
    def finish(self,raw,base,ext):
        out=base.copy()
        for c in ext:out[c]=ext[c].to_numpy()
        if self.nominal:out['distance_value_category']=raw['Flight Distance'].astype('int64').astype(str).to_numpy()
        for c in out.select_dtypes(include=['object','string','category']):out[c]=out[c].astype('object').fillna('__UNKNOWN__').astype(str)
        return out
    def fit_transform(self,raw,y,ext):
        self.base=DistanceFeatures('distance_stats_numeric')
        out=self.finish(raw,self.base.fit_transform_te(raw,y),ext)
        self.cat_columns=out.select_dtypes(include=['object','string']).columns.tolist()
        self.output_columns=list(out)
        assert out.shape[1]==(58 if self.nominal else 57)
        return out
    def transform(self,raw,ext):
        out=self.finish(raw,self.base.transform_te(raw),ext)
        assert list(out)==self.output_columns
        return out

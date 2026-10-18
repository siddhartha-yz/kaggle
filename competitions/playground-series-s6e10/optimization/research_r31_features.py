"""Same CatBoost numeric57 representation plus validated27 rating conditional fields."""
from research_features import DistanceFeatures
class ConditionalCatFeatures:
    def finish(self,base,ext,aux):
        out=base.copy()
        for c in ext:out[c]=ext[c].to_numpy()
        for c in aux:out[c]=aux[c].to_numpy('float32')
        for c in out.select_dtypes(include=['object','string','category']):out[c]=out[c].astype('object').fillna('__UNKNOWN__').astype(str)
        assert out.shape[1]==84
        return out
    def fit_transform(self,raw,y,ext,aux):
        self.base=DistanceFeatures('distance_stats_numeric');out=self.finish(self.base.fit_transform_te(raw,y),ext,aux)
        self.cat_columns=out.select_dtypes(include=['object','string']).columns.tolist();self.output_columns=list(out);return out
    def transform(self,raw,ext,aux):
        out=self.finish(self.base.transform_te(raw),ext,aux);assert list(out)==self.output_columns;return out

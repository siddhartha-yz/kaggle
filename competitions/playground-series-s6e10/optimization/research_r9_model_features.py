"""Existing 57 audited tree fields plus seven label-free-linkage teacher fields."""
import numpy as np
from research_features import DistanceFeatures
from research_r9_linkage_features import ExemplarLinkage

class LinkageFeatures:
    def __init__(self, source_raw, source_labels):
        self.source_raw = source_raw
        self.source_labels = np.asarray(source_labels,dtype='int8').copy()

    def finish(self, base, links, ext):
        out = base.copy()
        for c in ext: out[c] = ext[c].to_numpy()
        for c in links: out[c] = links[c].to_numpy()
        assert out.shape[1] == 64 and np.isfinite(links.to_numpy()).all()
        return out

    def fit_transform(self, raw, y, ext):
        self.base = DistanceFeatures('distance_stats_numeric')
        self.linkage = ExemplarLinkage().fit(raw,self.source_raw)
        self.source_raw = None
        links = self.linkage.transform(raw,self.source_labels)
        return self.finish(self.base.fit_transform_te(raw,y),links,ext)

    def transform(self, raw, ext):
        return self.finish(self.base.transform_te(raw),
                           self.linkage.transform(raw,self.source_labels),ext)

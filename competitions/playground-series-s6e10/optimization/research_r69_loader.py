"""Exact proper inner80/20 splits, own inner-training feature maps, no outer label fit."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline
from sklearn.model_selection import train_test_split
import numpy as np

def load_split(split):
    df,raw,y,outer,reference,tree,fold,aux=baseline(split);inner=[]
    for k,(fi,vi) in enumerate(outer):
        it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020)
        path=ROOT/f'research_artifacts/r69_rank_gate_{STAGE}/fold{k}/inner_indices.npz'
        z=np.load(path)
        assert all(np.array_equal(z[n],v) for n,v in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)])
        inner.append((it,im))
    return df,raw,y,inner,reference,tree,fold,aux

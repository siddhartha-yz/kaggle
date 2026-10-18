"""Exact currentv11 controls and unchangedNN74 covariates foruniformseedblocks."""
from research_r23_loader import ROOT,STAGE,pool,auxiliary,load_split as existing
import numpy as np

def load_split(split):
    df,raw,y,splits,_,tree,fold,aux=existing(split);nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz')
    assert np.array_equal(nn['id'],df.id) and np.array_equal(nn['target'],y) and np.array_equal(nn['fold'],fold)
    assert np.max(np.abs(pool(tree,nn['pred'])-nn['candidate']))<1e-12
    return df,raw,y,splits,nn['candidate'],tree,fold,aux

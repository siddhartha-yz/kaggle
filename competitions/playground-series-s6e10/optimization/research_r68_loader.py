"""Unchanged conditional74 inputs and sealed actualv11 reference probabilities."""
from research_r23_loader import ROOT,STAGE,pool,auxiliary,load_split as baseline
import numpy as np

def load_split(split):
    df,raw,y,splits,_,tree,fold,aux=baseline(split);best=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert all(np.array_equal(best[n],v) for n,v in [('id',df.id),('target',y),('fold',fold)])
    return df,raw,y,splits,best['candidate'],tree,fold,aux

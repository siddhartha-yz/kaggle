"""Unchanged old74 feature inputs and sealed actual-v11 controls."""
from research_r23_loader import ROOT,STAGE,pool,auxiliary,load_split as baseline
import numpy as np
def load_split(split):
    df,raw,y,splits,_,tree,fold,aux=baseline(split);best=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert np.array_equal(best['id'],df.id) and np.array_equal(best['target'],y) and np.array_equal(best['fold'],fold)
    return df,raw,y,splits,best['candidate'],tree,fold,aux

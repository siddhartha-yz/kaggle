"""Proper inner80/20 splits for new conditional74 neural expert fits only."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline
from sklearn.model_selection import train_test_split
import numpy as np,json
def load_split(split):
    df,raw,y,outer,reference,tree,fold,aux=baseline(split);inner=[]
    for k,(fi,vi) in enumerate(outer):
        it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);path=ROOT/f'research_artifacts/r57_nested_gate_{STAGE}/fold{k}/inner_indices.npz'
        if path.exists():
            z=np.load(path);assert np.array_equal(z['outer_train'],fi) and np.array_equal(z['outer_valid'],vi) and np.array_equal(z['inner_train'],it) and np.array_equal(z['inner_meta'],im)
        else:
            import sys
            assert int(sys.argv[sys.argv.index('--one-fold')+1])!=k
        inner.append((it,im))
    return df,raw,y,inner,reference,tree,fold,aux

"""Admitted v9 logpool control and fixed existing fold membership."""
from pathlib import Path
import sys,json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
STAGE='threefold' if '--threefold' in sys.argv else 'fivefold'
def pool(tree,neural):
    t=np.clip(np.asarray(tree,dtype='float64'),1e-6,1-1e-6)
    n=np.clip(np.asarray(neural,dtype='float64'),1e-6,1-1e-6)
    a=np.sqrt(t*n);b=np.sqrt((1-t)*(1-n));return a/(a+b)
def load_split(split):
    assert split=='oof' and json.loads((ROOT/'research_r14_result.json').read_text())['eligible_for_candidate']
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
    z=np.load(ROOT/f'research_artifacts/r14_log_pool_{STAGE}/oof.npz')
    assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
    assert np.max(np.abs(pool(z['tree'],z['neural'])-z['candidate']))<1e-12
    fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))]
    return df,raw,y,splits,z['candidate'],z['tree'],fold

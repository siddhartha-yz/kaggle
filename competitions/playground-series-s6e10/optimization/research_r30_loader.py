"""Exact formal-v11 NN74 inputs and tree controls; training maps refit per fold."""
from research_r15_loader import ROOT,STAGE,pool
from research_r23_loader import auxiliary
import numpy as np,pandas as pd

def load_split(split):
    assert split=='oof'
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
    z=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');t=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{STAGE}/oof.npz')
    assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and all(np.array_equal(z[k],t[k]) for k in ['id','target','fold'])
    assert np.max(np.abs(pool(t['tree'],z['pred'])-z['candidate']))<1e-12
    fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))]
    return df,raw,y,splits,z['candidate'],t['tree'],fold,auxiliary(df.id)

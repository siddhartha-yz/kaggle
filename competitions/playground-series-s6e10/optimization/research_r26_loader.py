"""Current formal v10 controls and hash-checked, ID-aligned target-free covariates."""
from research_r15_loader import ROOT,STAGE,pool
import json,numpy as np,pandas as pd

def auxiliary(ids,test=False):
    blocks=[];name='test' if test else 'train'
    for directory in ['r20_aux_officialcb','r21_conditional_officialcb','r25_posterior_officialcb']:
        src=ROOT/'research_artifacts'/directory;rec=json.loads((src/'result.json').read_text());assert rec['status']=='complete'
        z=np.load(src/f'{name}_features.npz');assert np.array_equal(z['id'],ids)
        from research_w3_common import sha
        assert sha(src/f'{name}_features.npz')==rec[f'{name}_features_sha256']
        blocks.append(pd.DataFrame(z['values'],columns=z['columns']))
    aux=pd.concat(blocks,axis=1);assert aux.shape==(len(ids),105) and aux.columns.is_unique and np.isfinite(aux.to_numpy()).all()
    return aux

def load_split(split):
    assert split=='oof'
    df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction'])
    z=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz')
    tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{STAGE}/oof.npz')
    assert all(np.array_equal(z[k],tree[k]) for k in ('id','target','fold'))
    assert np.max(np.abs(pool(tree['tree'],z['pred'])-z['candidate']))<1e-12
    assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
    fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))]
    return df,raw,y,splits,z['candidate'],tree['tree'],fold,auxiliary(df.id)

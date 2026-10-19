"""NN80 inputs; use only six feature-target probabilities from old source cache."""
from research_r15_loader import ROOT,STAGE,pool
from research_r23_loader import auxiliary as conditional
from research_w3_common import sha
import json,numpy as np,pandas as pd

def auxiliary(ids,test=False):
    base=conditional(ids,test=test);name='test' if test else 'train';src=ROOT/'research_artifacts/w3_service_teacher';rec=json.loads((src/'result.json').read_text());assert rec['status']=='complete' and rec['competition_targets_used_or_read'] is False
    z=np.load(src/f'{name}_features.npz');assert sha(src/f'{name}_features.npz')==rec[f'{name}_features_sha256'] and np.array_equal(z['id'],ids)
    columns=list(z['columns']);p=z['values'][:,[columns.index(f'svc_p{j}') for j in range(6)]]
    assert p.shape==(len(ids),6) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(1,dtype='float64')-1))<1e-6
    # Deliberately exclude source-SAT marginal, entropy/mean, and source lookup fields.
    return pd.concat([base,pd.DataFrame(p,columns=[f'source_missing_service_p{j}' for j in range(6)])],axis=1)

def load_split(split):
    assert split=='oof'
    plan=json.loads((ROOT/'research_r29_resource_plan.json').read_text());tree=np.load(ROOT/plan['tree_control'][STAGE])
    df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);formal=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz')
    assert np.array_equal(formal['id'],df.id) and np.array_equal(formal['target'],y) and all(np.array_equal(tree[k],formal[k]) for k in ['id','target','fold'])
    fold=formal['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))]
    return df,raw,y,splits,pool(tree['tree'],formal['pred']),tree['tree'],fold,auxiliary(df.id)

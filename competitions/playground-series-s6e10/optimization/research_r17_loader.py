"""Admitted v9 control plus independent original-LGB probabilities aligned by ID."""
from research_r15_loader import ROOT,STAGE,pool
import json,numpy as np,pandas as pd
def load_split(split):
    assert split=='oof' and json.loads((ROOT/'research_r14_result.json').read_text())['eligible_for_candidate']
    df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction'])
    z=np.load(ROOT/f'research_artifacts/r14_log_pool_{STAGE}/oof.npz')
    teacher=np.load(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')
    assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(teacher['id'],df.id)
    column=list(teacher['columns']).index('orig_teacher_lgb');prob=teacher['values'][:,column]
    fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))]
    return df,raw,y,splits,z['candidate'],z['tree'],fold,prob

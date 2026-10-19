"""Currentv11 controls; known27 and signedrank13 are ID/hash checked."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline,auxiliary as conditional
import json,numpy as np,pandas as pd
from research_w3_common import sha

def auxiliary(ids,test=False):
    src=ROOT/'research_artifacts/r37_ordinal_ranks';r=json.loads((src/'result.json').read_text());name='test' if test else 'train';z=np.load(src/f'{name}_features.npz');assert r['status']=='complete' and not r['satisfaction_column_loaded'] and r['new_auxiliary_models_fitted']==0 and np.array_equal(z['id'],ids) and sha(src/f'{name}_features.npz')==r[f'{name}_features_sha256']
    out=pd.concat([conditional(ids,test=test),pd.DataFrame(z['values'],columns=z['columns'])],axis=1);assert out.shape==(len(ids),40);return out

def load_split(split):
    df,raw,y,splits,_,tree,fold,_=baseline(split);nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert np.array_equal(nn['id'],df.id) and np.array_equal(nn['target'],y) and np.array_equal(nn['fold'],fold)
    return df,raw,y,splits,nn['candidate'],tree,fold,auxiliary(df.id)

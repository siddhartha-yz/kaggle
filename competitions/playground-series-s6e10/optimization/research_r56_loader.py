"""Two conditioning views at unchanged NN74 width; no new auxiliary model fits."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline,auxiliary as conditional
from research_w3_common import sha
import json,numpy as np,pandas as pd

def auxiliary(ids,test=False):
    src=ROOT/'research_artifacts/r46_cohort_auxiliary';r=json.loads((src/'result.json').read_text());name='test' if test else 'train';z=np.load(src/f'{name}_features.npz')
    assert r['status']=='complete' and not r['satisfaction_values_parsed'] and not r['satisfaction_used_as_input_or_target'] and r['all_thirteen_ratings_excluded_from_inputs']
    assert sha(src/f'{name}_features.npz')==r['features'][name]['sha256'] and np.array_equal(z['id'],ids)
    assert list(z['columns'][:13])==[f'cohort_rating_ev_{j}' for j in range(13)] and z['values'].shape==(len(ids),27)
    old=conditional(ids,test=test);out=pd.concat([pd.DataFrame(z['values'][:,:13],columns=z['columns'][:13]),old.iloc[:,13:]],axis=1)
    assert out.shape==(len(ids),27) and np.isfinite(out.to_numpy()).all() and np.array_equal(out.iloc[:,13:].to_numpy(),old.iloc[:,13:].to_numpy())
    return out

def load_split(split):
    df,raw,y,splits,_,tree,fold,_=baseline(split);nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert np.array_equal(nn['id'],df.id) and np.array_equal(nn['target'],y) and np.array_equal(nn['fold'],fold)
    return df,raw,y,splits,nn['candidate'],tree,fold,auxiliary(df.id)

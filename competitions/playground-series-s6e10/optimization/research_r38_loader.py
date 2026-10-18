"""ID/hash-checked currentv11 data with13label-freeordinalranks appended."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline,auxiliary as conditional
import json,numpy as np,pandas as pd
from research_w3_common import sha

def auxiliary(ids,test=False):
    src=ROOT/'research_artifacts/r37_ordinal_ranks';r=json.loads((src/'result.json').read_text());name='test' if test else 'train';z=np.load(src/f'{name}_features.npz');assert r['status']=='complete' and not r['satisfaction_column_loaded'] and r['new_auxiliary_models_fitted']==0 and np.array_equal(z['id'],ids) and sha(src/f'{name}_features.npz')==r[f'{name}_features_sha256']
    out=pd.concat([conditional(ids,test=test),pd.DataFrame(z['values'],columns=z['columns'])],axis=1);assert out.shape==(len(ids),40);return out

def load_split(split):
    df,raw,y,splits,reference,tree,fold,_=baseline(split)
    return df,raw,y,splits,reference,tree,fold,auxiliary(df.id)

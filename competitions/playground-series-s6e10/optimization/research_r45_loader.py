"""Certified27conditionalfieldsplusindependentnewsourceprobability,ID/hashchecked."""
from research_r23_loader import ROOT,STAGE,pool,auxiliary as conditional,load_split as baseline
from research_w3_common import sha
import json,numpy as np,pandas as pd

def auxiliary(ids,test=False):
    src=ROOT/'research_artifacts/r44_invistico_teacher';r=json.loads((src/'result.json').read_text());name='test' if test else 'train';z=np.load(src/f'{name}_features.npz')
    assert r['status']=='complete' and not r['official_satisfaction_loaded'] and r['provenance']['source_labels_only'] and r['provenance']['safe_rows']==129878 and np.array_equal(z['id'],ids) and sha(src/f'{name}_features.npz')==r['features'][name]['features_sha256']
    assert list(z['columns'])==['invistico_teacher_lgb'];out=pd.concat([conditional(ids,test=test),pd.DataFrame(z['values'],columns=z['columns'])],axis=1);assert out.shape==(len(ids),28);return out

def load_split(split):
    df,raw,y,splits,_,tree,fold,_=baseline(split);nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert np.array_equal(nn['id'],df.id) and np.array_equal(nn['target'],y) and np.array_equal(nn['fold'],fold)
    return df,raw,y,splits,nn['candidate'],tree,fold,auxiliary(df.id)

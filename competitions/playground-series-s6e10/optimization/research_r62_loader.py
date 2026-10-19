"""Old target-free27 plus13 original-only supervised likelihood-ratio fields."""
from research_r23_loader import ROOT,auxiliary as oldaux
from research_w3_common import sha
import json,numpy as np,pandas as pd
def auxiliary(ids,test=False):
    out=ROOT/'research_artifacts/r62_original_cohortclass';rec=json.loads((out/'result.json').read_text());name='test' if test else 'train';path=out/f'{name}_features.npz';z=np.load(path);assert rec['status']=='complete' and rec['official_SAT_values_parsed']==False and rec['external_original_SAT_conditioner'] and len(rec['models'])==13 and sha(path)==rec['features'][name]['sha256'] and np.array_equal(z['id'],ids) and list(z['columns'])==[f'original_cohort_class_llr_{j}' for j in range(13)]
    old=oldaux(ids,test=test);new=pd.DataFrame(z['values'],columns=z['columns']);assert np.isfinite(new.to_numpy()).all() and new.shape==(len(ids),13)
    return pd.concat([old,new],axis=1)

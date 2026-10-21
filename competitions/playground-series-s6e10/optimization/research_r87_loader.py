import numpy as np
from research_r23_loader import ROOT,auxiliary as current_auxiliary
from research_r87_features import TEACHER

def auxiliary(ids,test=False):
 out=current_auxiliary(ids,test=test);name='test' if test else 'train';z=np.load(ROOT/f'research_artifacts/original_teacher_v1/{name}_features.npz');assert np.array_equal(z['id'],ids)
 q=z['values'][:,list(z['columns']).index('orig_teacher_lgb')];assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
 out[TEACHER]=q.astype('float32');assert out.shape==(len(ids),28);return out

"""Evaluate the pre-frozen blend once on the fixed stability split."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from compare_validation import compare
root=Path(__file__).resolve().parent
recipe=json.loads((root/'frozen_recipe.json').read_text())
train=pd.read_csv(root/'data/train.csv');y=train.satisfaction.astype('int8')
_,valid=train_test_split(np.arange(len(train)),test_size=.2,random_state=recipe['confirm_split_seed'],stratify=y)
base=np.load(root/'iteration_artifacts/cbbase_confirm_s42/validation.npz')
assert np.array_equal(base['idx'],valid) and np.array_equal(base['target'],y.iloc[valid]) and np.array_equal(base['id'],train.id.iloc[valid])
pred=np.zeros(len(valid));components={}
for component in recipe['components']:
    folder=root/'iteration_artifacts'/f"{component['experiment']}_confirm_s42"
    r=json.loads((folder/'result.json').read_text()); v=np.load(folder/'validation.npz')
    assert r['status']=='complete' and r['fixed_trees']==r['best_iterations']==component['trees']
    assert np.array_equal(v['idx'],valid) and np.array_equal(v['target'],base['target']) and np.array_equal(v['id'],base['id'])
    assert np.isfinite(v['pred']).all() and ((v['pred']>=0)&(v['pred']<=1)).all()
    pred+=component['weight']*v['pred'];components[component['experiment']]=compare(base['target'],base['pred'],v['pred'])
result=compare(base['target'],base['pred'],pred)
result.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),components=components,split_seed=recipe['confirm_split_seed'],model_seed=42,weights_frozen_before_confirm=True,decision='proceed_full_frozen' if result['conditional_95pct_ci'][0]>0 else 'retain_baseline',caveat='Repeated official samples overlap development; this is a stability check, not an independent test. Conditional interval excludes selection uncertainty.')
primary=np.load(root/'artifacts/split_seed42.npz')['valid']
result['confirm_primary_validation_overlap_rows']=int(np.intersect1d(primary,valid).size)
with (root/'local_confirmation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))

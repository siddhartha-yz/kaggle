"""Assemble all three folds of one fixed view without interim quality."""
from research_r64_loader import ROOT,STAGE,VIEW,load_split
from research_w3_common import dump,sha,now
import json,numpy as np
assert STAGE=='threefold'
df,raw,y,splits,reference,tree,fold,aux=load_split('oof');out=ROOT/'research_artifacts'/f'r64_{VIEW}_{STAGE}';pred=np.full(len(y),np.nan)
for k,(fi,vi) in enumerate(splits):
    run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r64_execution_plan.json') and r['runner_sha256']==sha(ROOT/'research_r64_worker.py')
    for name,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('validation.npz','prediction_sha256')]:assert sha(run/name)==r[key]
    z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);pred[vi]=z['pred']
assert np.isfinite(pred).all()
np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=pred)
dump(out/'completed.json',dict(status='complete',completed_utc=now(),no_interim_quality=True,oof_sha256=sha(out/'oof.npz')))

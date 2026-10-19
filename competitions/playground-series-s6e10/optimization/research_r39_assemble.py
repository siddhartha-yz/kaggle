"""Uniformfourblock128memberaverage; old42predictions reused, no perseed scoring."""
from research_r4_common import ROOT,now,dump,sha
from research_r15_loader import pool
import json,numpy as np,sys

def main(stage):
    plan=json.loads((ROOT/'research_r39_plan.json').read_text());control=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');assert all(np.array_equal(control[k],tree[k]) for k in ['id','target','fold'])
    seed_predictions=[control['pred']];fold=control['fold']
    for seed in [43,44,45]:
        p=np.full(len(control['id']),np.nan)
        for k in sorted(np.unique(fold)):
            vi=np.flatnonzero(fold==k);run=ROOT/f'research_artifacts/r39_seed{seed}_lean_{stage}/fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['seed']==seed and r['config']==dict(plan['config'],random_state=seed) and r['runner_sha256']==plan['source_sha256']['research_r39_worker.py'] and r['plan_sha256']==sha(ROOT/'research_r39_plan.json') and r['model_sha256']==sha(run/'model.pkl') and r['transformer_sha256']==sha(run/'transformer.joblib')
            z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],control['id'][vi]) and np.array_equal(z['target'],control['target'][vi]);p[vi]=z['pred']
        assert np.isfinite(p).all();seed_predictions.append(p)
    ensemble=np.mean(seed_predictions,axis=0);candidate=pool(tree['tree'],ensemble);out=ROOT/f'research_artifacts/r39_uniform_{stage}';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'oof.npz',id=control['id'],target=control['target'],fold=fold,pred=ensemble,seed42=seed_predictions[0],seed43=seed_predictions[1],seed44=seed_predictions[2],seed45=seed_predictions[3],candidate=candidate,reference_v11=control['candidate'])
    dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True,seeds_uniform=[42,43,44,45],native_members_per_fold=128,old42_not_retrained=True))
if __name__=='__main__':main(sys.argv[1])

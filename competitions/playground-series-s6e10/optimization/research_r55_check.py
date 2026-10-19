"""Score the single randomized boosted-tree route only after all outer folds."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import numpy as np,json,sys

def check(stage):
    plan=json.loads((ROOT/'research_r55_plan.json').read_text())
    for name,digest in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/name)==digest,name
    out=ROOT/f'research_artifacts/r55_randomized_lgb84_{stage}';done=json.loads((out/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(out/'oof.npz')
    z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz')
    assert all(np.array_equal(z[n],base[n]) for n in ['id','target','fold'])
    assert np.array_equal(z['reference_v11'],base['candidate'])
    expected=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(z['pred'],1e-6,1-1e-6)))
    assert np.array_equal(z['candidate'],expected) and np.isfinite(expected).all()
    for k in sorted(np.unique(z['fold'])):
        run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());v=np.load(run/'validation.npz');vi=np.flatnonzero(z['fold']==k)
        assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r55_plan.json') and r['runner_sha256']==sha(ROOT/'research_r55_worker.py')
        assert r['prediction_sha256']==sha(run/'validation.npz') and r['model_sha256']==sha(run/'model.txt') and r['transformer_sha256']==sha(run/'transformer.joblib')
        assert np.array_equal(v['idx'],vi) and np.array_equal(v['id'],z['id'][vi]) and np.array_equal(v['target'],z['target'][vi]) and np.array_equal(v['pred'],z['pred'][vi])
    comparison=evidence(z['target'],base['candidate'],z['candidate'],z['fold'])
    result=dict(status='complete',checked_utc=now(),stage=stage,eligible=comparison['eligible_for_new_partition_replication'],vs_formal_v11=comparison,plan_sha256=sha(ROOT/'research_r55_plan.json'),oof_sha256=sha(out/'oof.npz'),all_folds_before_quality=True)
    dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':check(sys.argv[1])

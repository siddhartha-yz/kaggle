"""ranking-lossNN74 against actual formal/bestv11; priorv10 cumulative diagnostic only."""
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r15_loader import pool
import json,sys,numpy as np

def check(stage):
    out=ROOT/f'research_artifacts/r58_ranking_lean_{stage}';done=json.loads((out/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(out/'oof.npz')
    z=np.load(out/'oof.npz');control=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    assert all(np.array_equal(z[k],control[k]) and np.array_equal(z[k],tree[k]) for k in ('id','target','fold'))
    candidate=pool(tree['tree'],z['pred']);assert np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(z['reference_v11']-control['candidate']))<1e-12
    a=evidence(z['target'],control['candidate'],candidate,z['fold']);b=evidence(z['target'],tree['candidate'],candidate,z['fold'])
    r=dict(status='complete',stage=stage,checked_utc=now(),vs_formal_v11=a,vs_prior_v10_diagnostic=b,eligible=a['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r58_plan.json'))
    dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True);return r
if __name__=='__main__':check(sys.argv[1])

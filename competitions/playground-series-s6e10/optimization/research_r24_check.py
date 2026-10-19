"""One uncertainty route versus current formal/best v11; priorv10 cumulative diagnostic."""
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r15_loader import pool
import numpy as np,json,sys

def check(stage):
    out=ROOT/f'research_artifacts/r24_uncertainty_lgb_officialcb_{stage}';done=json.loads((out/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(out/'oof.npz')
    z=np.load(out/'oof.npz');control=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');old=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');n=control
    assert all(np.array_equal(z[k],control[k]) and np.array_equal(z[k],old[k]) and np.array_equal(z[k],n[k]) for k in ('id','target','fold'))
    assert np.max(np.abs(z['reference_v11']-control['candidate']))<1e-12 and np.max(np.abs(z['candidate']-pool(z['tree'],n['pred'])))<1e-12
    a=evidence(z['target'],control['candidate'],z['candidate'],z['fold']);b=evidence(z['target'],old['candidate'],z['candidate'],z['fold'])
    r=dict(status='complete',stage=stage,checked_utc=now(),eligible=a['eligible_for_new_partition_replication'],vs_formal_v11=a,vs_prior_v10_diagnostic=b,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r24_plan.json'))
    dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True);return r
if __name__=='__main__':check(sys.argv[1])

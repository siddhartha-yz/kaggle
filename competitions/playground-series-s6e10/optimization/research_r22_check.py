"""One frozen route vs current formal/best v10; reused-label conditional gates."""
from research_r4_common import ROOT,now,dump,sha,evidence
import numpy as np,json,sys
from sklearn.metrics import roc_auc_score

def check(stage):
    out=ROOT/f'research_artifacts/r22_global_tabicl_{stage}';done=json.loads((out/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(out/'oof.npz')
    z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    assert all(np.array_equal(z[k],base[k]) for k in ('id','target','fold')) and np.max(np.abs(z['reference_v10']-base['candidate']))<1e-12
    r=evidence(z['target'],base['candidate'],z['candidate'],z['fold']);r.update(status='complete',stage=stage,checked_utc=now(),eligible=bool(r['eligible_for_new_partition_replication']),single_tabicl_auc_diagnostic=float(roc_auc_score(z['target'],z['pred'])),oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r22_plan.json'))
    dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True);return r
if __name__=='__main__':check(sys.argv[1])

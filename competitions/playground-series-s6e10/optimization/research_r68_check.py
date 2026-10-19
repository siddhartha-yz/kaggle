"""One predeclared mixture, original dualbounds after all five/three folds."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import numpy as np,json,sys

def main(stage):
    out=ROOT/f'research_artifacts/r68_pure_ranknet_{stage}';done=json.loads((out/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert all(np.array_equal(z[n],base[n]) for n in ['id','target','fold'])
    candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(z['pred'],1e-6,1-1e-6)));assert np.max(np.abs(candidate-z['candidate']))<1e-12 and np.array_equal(z['reference_v11'],base['candidate']);proof=evidence(z['target'],base['candidate'],candidate,z['fold']);r=dict(status='complete',checked_utc=now(),stage=stage,vs_formal_v11=proof,eligible=proof['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r68_plan.json'));dump(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])

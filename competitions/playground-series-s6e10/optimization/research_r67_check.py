"""All outer folds completed before assessing the one fixed PairLogit expert mixture."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import numpy as np,json,sys

def main(stage):
    plan=json.loads((ROOT/'research_r67_plan.json').read_text());base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');out=ROOT/f'research_artifacts/r67_pairlogit84_{stage}';pred=np.full(len(base['id']),np.nan)
    for k in sorted(np.unique(base['fold'])):
        vi=np.flatnonzero(base['fold']==k);run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r67_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r67_worker.py']
        for n,h in r['file_sha256'].items():assert sha(run/n)==h,n
        z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],base['id'][vi]) and np.array_equal(z['target'],base['target'][vi]);pred[vi]=z['pred']
    assert np.isfinite(pred).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=candidate,reference_v11=base['candidate']);comparison=evidence(base['target'],base['candidate'],candidate,base['fold']);r=dict(status='complete',stage=stage,checked_utc=now(),plan_sha256=sha(ROOT/'research_r67_plan.json'),vs_formal_v11=comparison,eligible=comparison['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'),all_folds_before_outer_quality=True);dump(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])

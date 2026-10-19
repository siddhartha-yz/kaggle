"""Only assemble and score once every proper nested cohort gate has finished."""
from research_r4_common import ROOT,now,dump,sha,evidence
import numpy as np,json,sys
def main(stage):
    plan=json.loads((ROOT/'research_r65_plan.json').read_text());base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');out=ROOT/f'research_artifacts/r65_confidence_gate_{stage}';pred=np.full(len(base['id']),np.nan)
    for k in sorted(np.unique(base['fold'])):
        vi=np.flatnonzero(base['fold']==k);run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r65_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r65_gate_worker.py'] and r['validation_sha256']==sha(run/'validation.npz') and r['gate_sha256']==sha(run/'gate.joblib');z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and all(np.array_equal(z[n],base[n][vi]) for n in ['id','target']) and np.array_equal(z['reference_v11'],base['candidate'][vi]);pred[vi]=z['pred']
    assert np.isfinite(pred).all();comparison=evidence(base['target'],base['candidate'],pred,base['fold']);np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=pred,reference_v11=base['candidate']);r=dict(status='complete',checked_utc=now(),stage=stage,eligible=comparison['eligible_for_new_partition_replication'],vs_formal_v11=comparison,all_folds_before_quality=True,plan_sha256=sha(ROOT/'research_r65_plan.json'),oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])

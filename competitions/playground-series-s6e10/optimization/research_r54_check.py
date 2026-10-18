"""Only score after every fixed context in every fold has finished."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import json,numpy as np,sys
def main(stage):
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');fold=base['fold'];pred=np.full(len(fold),np.nan);out=ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}'
    for k in sorted(np.unique(fold)):
        vi=np.flatnonzero(fold==k);p=[]
        for m in range(4):
            run=out/f'fold{k}/context{m}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r54_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r54_worker.py'] and r['validation_sha256']==sha(run/'validation.npz');z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],base['id'][vi]) and np.array_equal(z['target'],base['target'][vi]);p.append(z['pred'])
        pred[vi]=np.mean(p,axis=0)
    assert np.isfinite(pred).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=fold,pred=pred,candidate=candidate,reference_v11=base['candidate']);comparison=evidence(base['target'],base['candidate'],candidate,fold);result=dict(status='complete',checked_utc=now(),stage=stage,eligible=comparison['eligible_for_new_partition_replication'],vs_formal_v11=comparison,plan_sha256=sha(ROOT/'research_r54_plan.json'),oof_sha256=sha(out/'oof.npz'),all_contexts_all_folds_before_quality=True);dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])

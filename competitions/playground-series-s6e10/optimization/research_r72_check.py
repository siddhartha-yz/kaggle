"""Complete4disjoint-context means, one frozen logit blend, current v11 gates."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,sys
from scipy.special import logit,expit

def combine(reference,pred):return expit(.875*logit(np.clip(reference.astype('float64'),1e-6,1-1e-6))+.125*logit(np.clip(pred.astype('float64'),1e-6,1-1e-6)))
def main(stage):
    plan=json.loads((ROOT/'research_r72_plan.json').read_text());out=ROOT/f'research_artifacts/r72_largecontext_raw21_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');old=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');assert all(np.array_equal(base[n],old[n]) for n in ['id','target','fold']);pred=np.full(len(base['id']),np.nan)
    for k in sorted(np.unique(base['fold'])):
        vi=np.flatnonzero(base['fold']==k);pv=np.zeros(len(vi));contexts=[]
        for j in range(4):
            run=out/f'fold{k}/context{j}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['runner_sha256']==plan['source_sha256']['research_r72_worker.py'] and r['plan_sha256']==sha(ROOT/'research_r72_plan.json') and r['prediction_sha256']==sha(run/'validation.npz') and r['context_sha256']==sha(run/'context.npz');q=np.load(run/'validation.npz');ctx=np.load(run/'context.npz');assert np.array_equal(q['idx'],vi) and np.array_equal(q['id'],base['id'][vi]) and np.array_equal(q['target'],base['target'][vi]) and not np.isin(ctx['idx'],vi).any();contexts.extend(ctx['idx'].tolist());pv+=q['pred']/4
        assert len(contexts)==len(set(contexts))==4*min(plan['context_rows'],int(np.sum(base['fold']!=k))//4);pred[vi]=pv
    assert np.isfinite(pred).all();candidate=combine(base['candidate'],pred);np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=candidate,reference_v12=base['candidate']);a=evidence(base['target'],base['candidate'],candidate,base['fold']);b=evidence(base['target'],old['candidate'],candidate,base['fold'])
    r=dict(status='complete',stage=stage,checked_utc=now(),eligible=a['eligible_for_new_partition_replication'],vs_formal_v12=a,vs_prior_v10_diagnostic=b,all_folds_before_scoring=True,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r72_plan.json'));dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

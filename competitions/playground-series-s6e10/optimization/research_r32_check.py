"""Assess one frozen logit blend only after every fold completes; diagnostic old v10."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import json,numpy as np,sys

def combine(tree,nn,tabm):
    return expit(.5*logit(np.clip(tree,1e-6,1-1e-6))+.375*logit(np.clip(nn,1e-6,1-1e-6))+.125*logit(np.clip(tabm,1e-6,1-1e-6)))
def main(stage):
    plan=json.loads((ROOT/'research_r32_plan.json').read_text());out=ROOT/f'research_artifacts/r32_tabm74_{stage}'
    nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');assert all(np.array_equal(nn[k],tree[k]) for k in ['id','target','fold'])
    p=np.full(len(nn['id']),np.nan)
    for k in sorted(np.unique(nn['fold'])):
        run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['runner_sha256']==plan['source_sha256']['research_r32_worker.py'] and r['model_sha256']==sha(run/'model.pt');v=np.load(run/'validation.npz');vi=np.flatnonzero(nn['fold']==k);assert np.array_equal(v['idx'],vi) and np.array_equal(v['id'],nn['id'][vi]) and np.array_equal(v['target'],nn['target'][vi]);p[vi]=v['pred']
    assert np.isfinite(p).all();candidate=combine(tree['tree'],nn['pred'],p)
    np.savez_compressed(out/'oof.npz',id=nn['id'],target=nn['target'],fold=nn['fold'],pred=p,candidate=candidate,reference_v11=nn['candidate'])
    a=evidence(nn['target'],nn['candidate'],candidate,nn['fold']);b=evidence(nn['target'],tree['candidate'],candidate,nn['fold'])
    r=dict(status='complete',stage=stage,checked_utc=now(),eligible=a['eligible_for_new_partition_replication'],vs_formal_v11=a,vs_prior_v10_diagnostic=b,all_folds_before_scoring=True,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r32_plan.json'))
    dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

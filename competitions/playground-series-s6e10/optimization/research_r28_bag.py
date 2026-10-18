"""One frozen equal-probability main-booster average; no new fivefold fits."""
from research_r4_common import ROOT,now,dump,sha
from research_r15_loader import pool
import json,numpy as np

def main():
    plan=json.loads((ROOT/'research_r28_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    src=ROOT/'research_artifacts/r27_xgb_fivefold';done=json.loads((src/'completed.json').read_text());assert done['status']=='complete' and done['oof_sha256']==sha(src/'oof.npz')
    x=np.load(src/'oof.npz');base=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_fivefold/oof.npz');nn=np.load(ROOT/'research_artifacts/r23_conditional_lean_fivefold/oof.npz')
    assert all(np.array_equal(x[k],base[k]) and np.array_equal(x[k],nn[k]) for k in ['id','target','fold'])
    assert np.max(np.abs(pool(base['tree'],nn['pred'])-nn['candidate']))<1e-12
    tree=base['tree']+.234375*(x['pred']-base['pred']);candidate=pool(tree,nn['pred']);assert ((tree>=0)&(tree<=1)).all() and np.isfinite(candidate).all()
    out=ROOT/'research_artifacts/r28_bag_fivefold';out.mkdir(exist_ok=True);np.savez_compressed(out/'oof.npz',id=x['id'],target=x['target'],fold=x['fold'],pred=x['pred'],tree=tree,candidate=candidate,reference_v11=nn['candidate'])
    dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_new_fivefold_model_fits=True,no_other_blendweights_evaluated=True))
if __name__=='__main__':main()

"""Evaluate exactly one frozen meanfour density family after all folds, versus actual v12."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,sys
from scipy.special import expit,logit

def main(stage):
 plan=json.loads((ROOT/'research_r73_plan.json').read_text());out=ROOT/f'research_artifacts/r73_joint_density_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');pred=np.full(len(base['id']),np.nan)
 for k in sorted(np.unique(base['fold'])):
  run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r73_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r73_worker.py'] and r['prediction_sha256']==sha(run/'validation.npz') and r['fit_indices_sha256']==sha(run/'fit_indices.npz') and r['encoder_sha256']==sha(run/'encoder.json') and all(sha(run/n)==h for n,h in r['member_files_sha256'].items());vi=np.flatnonzero(base['fold']==k);fi=np.flatnonzero(base['fold']!=k);z=np.load(run/'validation.npz');fit=np.load(run/'fit_indices.npz');assert np.array_equal(fit['idx'],fi) and np.array_equal(fit['id'],base['id'][fi]);assert all(np.array_equal(z[n],value) for n,value in [('idx',vi),('id',base['id'][vi]),('target',base['target'][vi])]);parts=[]
  for j in range(4):
   p=np.load(run/f'member{j}.npz');assert np.array_equal(p['idx'],vi) and np.array_equal(p['id'],base['id'][vi]);parts.append(p['pred'])
  assert np.array_equal(np.mean(np.stack(parts),axis=0),z['pred']);pred[vi]=z['pred']
 assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=candidate,reference_v12=base['candidate']);result=evidence(base['target'],base['candidate'],candidate,base['fold']);record=dict(status='complete',stage=stage,checked_utc=now(),eligible=result['eligible_for_new_partition_replication'],vs_formal_v12=result,all_folds_before_scoring=True,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r73_plan.json'));dump(out/'result.json',record);print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

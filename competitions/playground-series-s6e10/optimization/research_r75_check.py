"""Allfolds-first fixed TabDPT/v12 probability pool and original dual bounds."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,sys
from scipy.special import expit,logit
def main(stage):
 plan=json.loads((ROOT/'research_r75_plan.json').read_text());out=ROOT/f'research_artifacts/r75_tabdpt_raw21_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');pred=np.full(len(base['id']),np.nan)
 for k in sorted(np.unique(base['fold'])):
  run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r75_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r75_worker.py'] and r['pretrained_weights_unchanged'] and r['pretrained_parameters_sha256']==plan['pretrained_parameters_sha256'] and all(sha(run/n)==h for n,h in r['files_sha256'].items());fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);ctx=np.load(run/'contexts.npz');assert all(np.array_equal(ctx[n],v) for n,v in [('fi',fi),('vi',vi),('train_id',base['id'][fi]),('valid_id',base['id'][vi])]);z=np.load(run/'validation.npz');assert all(np.array_equal(z[n],v) for n,v in [('idx',vi),('id',base['id'][vi]),('target',base['target'][vi])])
  for start in range(0,len(vi),4096):
   q=np.load(run/f'query{start:06d}.npz');take=vi[start:start+4096];assert str(q['plan_sha256'])==sha(ROOT/'research_r75_plan.json') and np.array_equal(q['idx'],take) and np.array_equal(q['id'],base['id'][take]) and np.array_equal(q['pred'],z['pred'][start:start+len(take)])
  pred[vi]=z['pred']
 assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=candidate,reference_v12=base['candidate']);proof=evidence(base['target'],base['candidate'],candidate,base['fold']);record=dict(status='complete',stage=stage,checked_utc=now(),eligible=proof['eligible_for_new_partition_replication'],vs_formal_v12=proof,all_folds_before_scoring=True,pretrained_weight_updates=False,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(ROOT/'research_r75_plan.json'));dump(out/'result.json',record);print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

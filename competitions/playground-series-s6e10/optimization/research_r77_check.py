"""All folds first; one frozen1/16 EBM versus actualv12 logit fusion."""
from research_r4_common import ROOT,now,dump,sha,evidence
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit,logit
from interpret.glassbox._ebm._utils import make_bag
import json,sys,numpy as np

def main(stage):
 plan=json.loads((ROOT/'research_r77_plan.json').read_text());ph=sha(ROOT/'research_r77_plan.json');out=ROOT/f'research_artifacts/r77_ebm_raw21_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');pred=np.full(len(base['id']),np.nan);K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(pred),-1,dtype='int8')
 for j,(_,ii) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros((len(pred),1)),base['target'])):fold[ii]=j
 assert np.array_equal(fold,base['fold'])
 for k in range(K):
  run=out/f'fold{k}';d=json.loads((run/'result.json').read_text());fit=json.loads((run/'training.json').read_text());assert d['status']=='complete' and d['plan_sha256']==ph and fit['plan_sha256']==ph and d['runner_sha256']==plan['source_sha256']['research_r77_worker.py'] and not fit['outer_validation_used'] and fit['parameters']==plan['parameters'] and fit['backend']=='ThreadingBackend' and all(sha(run/n)==h for n,h in d['files_sha256'].items());assert max(d['model_reload_error'],d['additive_formula_error'],d['query_batch_error'])<1e-12
  fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);z=np.load(run/'outer_indices.npz');assert all(np.array_equal(z[n],v) for n,v in [('fi',fi),('vi',vi),('train_id',base['id'][fi]),('valid_id',base['id'][vi])]);v=np.load(run/'validation.npz');assert all(np.array_equal(v[n],a) for n,a in [('idx',vi),('id',base['id'][vi]),('target',base['target'][vi])])
  bags=np.load(run/'inner_bags.npz');assert bags['bag_matrix'].shape==(8,len(fi)) and fit['inner_bags_sha256']==sha(run/'inner_bags.npz')
  for j in range(8):
   rng=bags['rng_before'][j].copy();original=make_bag(base['target'][fi].astype('int64'),2,.15,rng,True);assert np.array_equal(original,bags['bag_matrix'][j]) and np.array_equal(rng,bags['rng_after'][j])
  for start in range(0,len(vi),16384):
   q=np.load(run/f'query{start:06d}.npz');ii=vi[start:start+16384];assert str(q['plan_sha256'])==ph and np.array_equal(q['idx'],ii) and np.array_equal(q['id'],base['id'][ii]) and np.array_equal(q['pred'],v['pred'][start:start+len(ii)])
  pred[vi]=v['pred']
 assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=expit(.9375*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.0625*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=fold,pred=pred,candidate=candidate,reference_v12=base['candidate']);proof=evidence(base['target'],base['candidate'],candidate,fold);d=dict(status='complete',stage=stage,checked_utc=now(),eligible=proof['eligible_for_new_partition_replication'],vs_formal_v12=proof,all_folds_before_scoring=True,plan_sha256=ph,oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',d);print(json.dumps(d,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

"""Evaluate one fixed representation only after all outer folds are complete."""
from research_r78_model import ROOT,sha,references,candidate,np
from research_r4_common import evidence,now,dump
from sklearn.model_selection import StratifiedKFold
import pandas as pd,json,sys
def main(stage):
 plan=json.loads((ROOT/'research_r78_plan.json').read_text());ph=sha(ROOT/'research_r78_plan.json');base,tree=references(stage);frame=pd.read_csv(ROOT/'data/train.csv');y=frame.satisfaction.to_numpy('int8');ids=frame.id.to_numpy();K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(y),-1,'int8');pred=np.full(len(y),np.nan);out=ROOT/f'research_artifacts/r78_logratio_{stage}'
 assert np.array_equal(y,base['target']) and np.array_equal(ids,base['id'])
 for k,(fi,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
  fold[vi]=k;run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==ph and r['config']==plan['config'] and r['feature_width']==139 and not r['validation_used_for_fit_or_epoch_selection'] and r['original74_query_inputs_exact'];assert all(sha(run/n)==h for n,h in r['files_sha256'].items());assert len(r['epochs'])==3 and r['model_reload_error']<1e-7 and r['peak_GPU_bytes']<=plan['resource_limits']['own_GPU_ceiling_bytes'] and r['peak_RSS_bytes']<=plan['resource_limits']['own_RSS_ceiling_bytes']
  assert all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=plan['resource_limits']['allocator_cap_bytes'] for e in r['epochs']);ix=np.load(run/'indices.npz');z=np.load(run/'validation.npz')
  for n,v in [('fi',fi),('vi',vi),('train_id',ids[fi]),('valid_id',ids[vi])]:assert np.array_equal(ix[n],v)
  assert all(np.array_equal(z[n],v) for n,v in [('idx',vi),('id',ids[vi]),('target',y[vi])]);pred[vi]=z['pred']
 assert np.array_equal(fold,base['fold']) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();p=candidate(tree,pred,base['pred']);np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred,candidate=p,reference_v12=base['candidate']);proof=evidence(y,base['candidate'],p,fold);r=dict(status='complete',checked_utc=now(),stage=stage,eligible=proof['eligible_for_new_partition_replication'],vs_formal_v12=proof,all_folds_before_scoring=True,candidates=1,plan_sha256=ph,oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

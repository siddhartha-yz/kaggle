"""Score exactly one prospective TabSTAR pool only after all outerfits finish."""
from research_r4_common import ROOT,sha,dump,evidence,now
from sklearn.model_selection import StratifiedKFold
from tabstar.preprocessing.splits import split_to_val
from scipy.special import expit,logit
import pandas as pd,numpy as np,json,sys
def main(stage):
 plan=json.loads((ROOT/'research_r79r2_plan.json').read_text());ph=sha(ROOT/'research_r79r2_plan.json');out=ROOT/f'research_artifacts/r79r2_tabstar_raw21_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');frame=pd.read_csv(ROOT/'data/train.csv');y=frame.satisfaction.to_numpy('int8');ids=frame.id.to_numpy();raw=frame.drop(columns=['id','satisfaction']);labels=pd.Series(np.where(y,'True','False'),name='satisfaction');assert np.array_equal(y,base['target']) and np.array_equal(ids,base['id']);K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(y),-1,dtype='int8');pred=np.full(len(y),np.nan)
 for k,(fi,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):
  fold[vi]=k;run=out/f'fold{k}';d=json.loads((run/'result.json').read_text());f=json.loads((run/'training.json').read_text());assert d['status']=='complete' and d['plan_sha256']==f['plan_sha256']==ph and all(sha(run/n)==h for n,h in d['files_sha256'].items()) and not f['outer_validation_used_for_fit'];assert f['parameters']==plan['parameters'] and f['pretrained_base_parameters_sha256']==plan['pretrained_base_parameters_sha256'] and len(f['trace']['epochs'])==len(f['trace']['inner_evaluations'])==3 and d['reload_error']<1e-6 and d['batch_error']<.002
  for n,h in f['native_best_checkpoint_files'].items():assert sha(run/n)==h
  ix=np.load(run/'indices.npz');X,I,Y,J=split_to_val(raw.iloc[fi],labels.iloc[fi],is_cls=True)
  for n,v in [('fi',fi),('vi',vi),('train_id',ids[fi]),('valid_id',ids[vi]),('inner_train',X.index),('inner_valid',I.index),('inner_train_id',ids[X.index]),('inner_valid_id',ids[I.index])]:assert np.array_equal(ix[n],v)
  saved=np.load(run/'validation.npz');assert all(np.array_equal(saved[n],v) for n,v in [('idx',vi),('id',ids[vi]),('target',y[vi])])
  for start in range(0,len(vi),16384):
   q=np.load(run/f'query{start:06d}.npz');ii=vi[start:start+16384];assert str(q['plan_sha256'])==ph and np.array_equal(q['idx'],ii) and np.array_equal(q['id'],ids[ii]) and np.array_equal(q['pred'],saved['pred'][start:start+len(ii)])
  pred[vi]=saved['pred']
 assert np.array_equal(fold,base['fold']) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
 candidate=expit(.9375*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.0625*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred,candidate=candidate,reference_v12=base['candidate']);proof=evidence(y,base['candidate'],candidate,fold);d=dict(status='complete',checked_utc=now(),stage=stage,eligible=proof['eligible_for_new_partition_replication'],vs_formal_v12=proof,all_folds_before_scoring=True,candidates=1,plan_sha256=ph,oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',d);print(json.dumps(d,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

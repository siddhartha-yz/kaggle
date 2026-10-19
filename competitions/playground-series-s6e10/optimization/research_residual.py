"""Refine an independent original teacher as a fixed logit prior, not just an input."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,argparse,signal,gc
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from research_features import DistanceFeatures
from research_external_ablation import external
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','confirm','oof'],default='primary');p.add_argument('--trees',type=int,default=2200);p.add_argument('--early-stop',action='store_true');a=p.parse_args();assert not a.early_stop or a.split=='primary'
 protocol=json.loads((ROOT/'research_protocol.json').read_text());deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))
 name=f'residual_prior05_{a.split}_t{a.trees}'+('_es' if a.early_stop else '');out=ROOT/'research_artifacts'/name;out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.astype('int8').to_numpy();ext,_=external('both',df.id);prior=.5*logit(np.clip(ext.orig_teacher_lgb.to_numpy(),1e-6,1-1e-6))
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];ref=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz')['pred']
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])];ref=np.load(ROOT/'research_artifacts'/f'e1_original_both_{a.split}'/'validation.npz')['pred']
 params=dict(n_estimators=a.trees,learning_rate=.04,num_leaves=63,max_depth=-1,min_child_samples=150,reg_lambda=10,objective='binary',metric='auc',n_jobs=6,random_state=42,verbosity=-1,colsample_bytree=1.,subsample=1.,max_bin=255,deterministic=True,force_col_wise=True)
 plan={'params':params,'prior_scale':.5,'prior_source':'Independent CC0 original-only teacher; exact overlaps excluded','early_stopping_primary_only':a.early_stop,'runner_sha256':sha(__file__),'preprocessing_sha256':sha(ROOT/'research_features.py'),'source_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'frozen_utc':datetime.now(timezone.utc).isoformat()}
 if not (out/'plan.json').exists():dump(out/'plan.json',plan)
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc']);start=time.monotonic();rec={'status':'training','fold':k,'started_utc':datetime.now(timezone.utc).isoformat()};dump(run/'result.json',rec)
  try:
   tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
   for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
   kwargs={'init_score':prior[fi]}
   if a.early_stop:kwargs.update(eval_set=[(V,y[vi])],eval_init_score=[prior[vi]],eval_metric='auc',callbacks=[lgb.early_stopping(150,first_metric_only=True),lgb.log_evaluation(200)])
   model=lgb.LGBMClassifier(**params);model.fit(X,y[fi],**kwargs);best=int(model.best_iteration_ or a.trees);pred=expit(model.booster_.predict(V,raw_score=True,num_iteration=best,num_threads=6)+prior[vi]);auc=float(roc_auc_score(y[vi],pred))
   if a.early_stop:assert abs(auc-model.best_score_['valid_0']['auc'])<1e-12, 'Prior addition must match LightGBM evaluation'
   model.booster_.save_model(str(run/'model.txt'),num_iteration=best);loaded=lgb.Booster(model_file=str(run/'model.txt'));again=expit(loaded.predict(V,raw_score=True,num_threads=6)+prior[vi]);assert np.allclose(pred,again,rtol=0,atol=1e-14)
   np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);rec.update(status='complete',auc=auc,selected_trees=best,model_sha256=sha(run/'model.txt'),reload_maxdiff=float(np.max(np.abs(pred-again))),elapsed_seconds=time.monotonic()-start);dump(run/'result.json',rec);print(name,k,rec,flush=True);del X,V,model;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e));dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  target=y;ids=df.id.to_numpy();idx=np.arange(len(y));np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred)
 else:z=np.load(out/'fold0/validation.npz');pred=z['pred'];idx=z['idx'];target=y[idx];ids=df.id.iloc[idx].to_numpy()
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred);comparison=compare(target,ref,pred);result={'status':'complete','name':name,'auc':float(roc_auc_score(target,pred)),'vs_original_lgb':comparison,'fixed_grid_blends':{str(w):compare(target,ref,(1-w)*ref+w*pred) for w in [.25,.5]},'selected_trees':[json.loads((out/f'fold{k}/result.json').read_text())['selected_trees'] for k in range(len(splits))],'completed_utc':datetime.now(timezone.utc).isoformat()}
 if a.split=='oof':result['byfold']=[compare(target[fold==k],ref[fold==k],pred[fold==k]) for k in range(5)]
 dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

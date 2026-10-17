"""Frozen-feature model variants on existing development or fixed OOF rows."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,argparse,signal,gc
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score,log_loss
from research_features import DistanceFeatures
from research_external_ablation import external
from research_oof import params
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['lgb','cb','xgb'],required=True);p.add_argument('--external',choices=['none','teacher','lookup','both','both_neighbors','both_xgb'],default='none');p.add_argument('--split',choices=['primary','confirm','oof'],required=True);p.add_argument('--trees',type=int,default=2000);p.add_argument('--early-stop',action='store_true');p.add_argument('--representation',choices=['stats','distribution','interactions'],default='stats');p.add_argument('--profile',choices=['base','wide'],default='base');a=p.parse_args()
 assert a.profile!='wide' or a.kind=='lgb'
 assert not a.early_stop or (a.split=='primary' and (a.kind=='xgb' or a.profile=='wide')), 'Early stopping only on designated development selection'
 protocol=json.loads((ROOT/'research_protocol.json').read_text())
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
 name=f'{a.kind}_stats_orig_{a.external}_{a.split}'+(f'_t{a.trees}'+('_es' if a.early_stop else '') if a.kind=='xgb' else '')+('_'+a.representation if a.representation!='stats' else '')+(f'_wide_t{a.trees}'+('_es' if a.early_stop else '') if a.profile=='wide' else '');out=ROOT/'research_artifacts'/name;out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction'])
 ext,matched=(None,None) if a.external=='none' else external(a.external,df.id)
 model_params=params('cb1308' if a.kind=='cb' else 'stats639')
 if a.profile=='wide':model_params.update(n_estimators=a.trees,learning_rate=.02,num_leaves=127,min_child_samples=100,reg_lambda=1,colsample_bytree=.5,subsample=.8,subsample_freq=1,metric='auc')
 if a.kind=='xgb':
  model_params=dict(n_estimators=a.trees,learning_rate=.04,max_depth=6,min_child_weight=20,reg_lambda=20,subsample=.8,colsample_bytree=.9,max_bin=256,objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cpu',enable_categorical=True,n_jobs=6,random_state=42)
  if a.early_stop:model_params['early_stopping_rounds']=150
 plan={'name':name,'kind':a.kind,'external':a.external,'split':a.split,'params':model_params,'preprocessing_sha256':sha(ROOT/'research_features.py'),'runner_sha256':sha(__file__),'external_sha256':None if ext is None else sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'frozen_utc':now()}
 if a.representation=='distribution':plan['distribution_preprocessing_sha256']=sha(ROOT/'research_distributions.py')
 if a.representation=='interactions':plan['inner_oof_interactions']={'alpha':20,'keys':[['Flight Distance'],['Age'],['Flight Distance','Age'],['Flight Distance','Class'],['Flight Distance','Type of Travel'],['Age','Class','Type of Travel','Customer Type']]}
 if a.external=='both_xgb':plan['xgb_teacher_sha256']=sha(ROOT/'research_artifacts/original_xgb_v1/train_features.npz')
 if a.external=='both_neighbors':plan['neighbors_sha256']=sha(ROOT/'research_artifacts/original_neighbors_v1/train_features.npz')
 if (out/'plan.json').exists():
  old=json.loads((out/'plan.json').read_text());assert all(old[k]==plan[k] for k in ['name','params','preprocessing_sha256','external_sha256']);assert all(old.get(k)==plan.get(k) for k in ['neighbors_sha256','distribution_preprocessing_sha256','inner_oof_interactions','xgb_teacher_sha256'])
 else:dump(out/'plan.json',plan)
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)]
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])]
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
  start=time.monotonic();rec={'status':'training','fold':k,'started_utc':now(),'train_rows':len(fi),'valid_rows':len(vi)};dump(run/'result.json',rec)
  try:
   if a.representation=='distribution':
    from research_distributions import DistanceDistributions
    tr=DistanceDistributions()
   elif a.representation=='interactions':tr=DistanceFeatures('distance_stats_numeric',te_columns=plan['inner_oof_interactions']['keys'],alpha=20.)
   else:tr=DistanceFeatures('distance_stats_numeric')
   X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
   rec['te_audit']=tr.te_audit
   if ext is not None:
    for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
   if a.kind=='cb':
    cats=X.select_dtypes('category').columns.tolist()
    for c in cats:X[c]=X[c].astype(str);V[c]=V[c].astype(str)
    model=CatBoostClassifier(**model_params);model.fit(X,y[fi],cat_features=cats);mf='model.cbm';model.save_model(str(run/mf))
   elif a.kind=='xgb':
    from xgboost import XGBClassifier
    model=XGBClassifier(**model_params);model.fit(X,y[fi],**({'eval_set':[(V,y[vi])],'verbose':200} if a.early_stop else {}));mf='model.json';model.save_model(str(run/mf));rec['selected_trees']=int(model.best_iteration+1) if a.early_stop else a.trees
   else:
    kwargs={'eval_set':[(V,y[vi])],'eval_metric':'auc','callbacks':[lgb.early_stopping(150,first_metric_only=True),lgb.log_evaluation(200)]} if a.early_stop else {}
    model=lgb.LGBMClassifier(**model_params);model.fit(X,y[fi],**kwargs);mf='model.txt';model.booster_.save_model(str(run/mf));rec['selected_trees']=int(model.best_iteration_ or model_params['n_estimators'])
   pred=model.predict_proba(V)[:,1];assert np.isfinite(pred).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
   rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),model_file=mf,model_sha256=sha(run/mf),elapsed_seconds=time.monotonic()-start,ended_utc=now());dump(run/'result.json',rec);print(name,k,rec['auc'],rec['elapsed_seconds'],flush=True);del model,X,V;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e));dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):
   z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['target'],y[vi]);pred[vi]=z['pred']
  assert np.isfinite(pred).all();target=y;ids=df.id.to_numpy();idx=np.arange(len(y));base=np.load(ROOT/'research_artifacts/oof_v1/stats639_oof.npz')['pred'];np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred)
  byfold=[compare(y[fold==k],base[fold==k],pred[fold==k]) for k in range(5)]
 else:
  idx=splits[0][1];z=np.load(out/'fold0/validation.npz');pred=z['pred'];target=y[idx];ids=df.id.iloc[idx].to_numpy();z=np.load(ROOT/'research_artifacts'/f'b1_distance_stats_numeric_{a.split}'/'validation.npz');assert np.array_equal(z['idx'],idx);base=z['pred'];byfold=[]
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred)
 comparison=compare(target,base,pred);result={'status':'complete','name':name,'auc':float(roc_auc_score(target,pred)),'logloss':float(log_loss(target,pred)),'vs_stats':comparison,'byfold':byfold,'completed_utc':now()}
 if a.split=='oof':result['stable_positive_vs_stats']=all(c['delta_auc']>0 for c in byfold) and comparison['conditional_95pct_ci'][0]>0
 if a.external in ['both_neighbors','both_xgb'] or a.representation!='stats' or a.profile=='wide':
  refpath=ROOT/'research_artifacts'/('lgb_stats_orig_both_oof/oof.npz' if a.split=='oof' else f'e1_original_both_{a.split}/validation.npz');old=np.load(refpath);assert np.array_equal(old['id'],ids) and np.array_equal(old['target'],target);op=old['pred'];c=compare(target,op,pred);result['vs_original_lgb']=c
  if a.split=='oof':
   bf=[compare(target[fold==k],op[fold==k],pred[fold==k]) for k in range(5)];result['byfold_vs_original_lgb']=bf;result['promote_alternative']=all(z['delta_auc']>0 for z in bf) and c['delta_auc']>=.0002 and c['conditional_95pct_ci'][0]>0
 if matched is not None:result['slices']={n:{'rows':int(m.sum()),'comparison':compare(target[m],base[m],pred[m])} for n,m in [('near_match',matched[idx]),('no_near_match',~matched[idx])]}
 dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

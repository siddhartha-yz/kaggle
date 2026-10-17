"""Fixed five-fold, train-fold-only preprocessing; sequential CPU fits and resumable outputs."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, time, signal, argparse, gc
os.environ['OMP_NUM_THREADS']='6'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from research_features import DistanceFeatures
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts'/'oof_v1'
MODELS=['stats639','base639','bag639','cb1308']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); t.replace(p)
def now(): return datetime.now(timezone.utc).isoformat()
def check_time():
 p=json.loads((ROOT/'research_protocol.json').read_text())
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc']), 'No new training after cutoff'
def timeout(signum,frame): raise TimeoutError('Training hard cutoff reached')
def params(name):
 if name=='cb1308': return dict(iterations=1308,learning_rate=.07,depth=8,l2_leaf_reg=5,loss_function='Logloss',eval_metric='AUC',random_seed=42,task_type='CPU',thread_count=6,allow_writing_files=False,verbose=200)
 if name=='bag639': return json.loads((ROOT/'bagging_plan.json').read_text())['parameters']
 return dict(n_estimators=639,learning_rate=.04,num_leaves=63,max_depth=-1,min_child_samples=150,reg_lambda=10,objective='binary',n_jobs=6,random_state=42,verbosity=-1,colsample_bytree=1.,subsample=1.,max_bin=255,deterministic=True,force_col_wise=True)
def main():
 a=argparse.ArgumentParser(); a.add_argument('--models',nargs='+',choices=MODELS,default=MODELS); a=a.parse_args()
 OUT.mkdir(exist_ok=True)
 protocol=json.loads((ROOT/'research_protocol.json').read_text())
 remaining=(datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds()
 assert remaining>0; signal.signal(signal.SIGALRM,timeout); signal.alarm(int(remaining))
 train=pd.read_csv(ROOT/'data/train.csv'); raw=train.drop(columns=['id','satisfaction']); y=train.satisfaction.astype('int8').to_numpy()
 assert sha(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'
 fold=np.full(len(y),-1,dtype='int8')
 for i,(_,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261018).split(raw,y)): fold[vi]=i
 splits=OUT/'folds.npz'
 if splits.exists():
  saved=np.load(splits); assert np.array_equal(saved['fold'],fold) and np.array_equal(saved['id'],train.id)
 else: np.savez_compressed(splits,fold=fold,id=train.id.to_numpy(),target=y)
 frozen={'seed':20261018,'folds':5,'fold_file_sha256':sha(splits),'trees_fixed_before_oof':True,'early_stopping':False,'models':{n:params(n) for n in MODELS},'frozen_utc':now(),'feature_code_sha256':sha(ROOT/'research_features.py'),'runner_sha256':sha(__file__),'comparison':'Current submitted blend fixed .4 cb1308 + .4 base639 + .2 bag639 versus stats639; candidate promotion requires positive delta in every fold and positive pooled paired conditional 95% lower bound. No blend search in this comparison.','limitation':'Hyperparameters/features were selected on earlier overlapping development splits; OOF is a stability estimate, not an untouched unbiased final holdout.'}
 if not (OUT/'plan.json').exists(): dump(OUT/'plan.json',frozen)
 else:
  old=json.loads((OUT/'plan.json').read_text()); assert old['feature_code_sha256']==frozen['feature_code_sha256'] and old['models']==frozen['models'] and old['fold_file_sha256']==frozen['fold_file_sha256']
 for name in a.models:
  for k in range(5):
   run=OUT/f'{name}_fold{k}'; run.mkdir(exist_ok=True); result=run/'result.json'
   if result.exists() and json.loads(result.read_text()).get('status')=='complete':
    rec=json.loads(result.read_text()); assert rec['model_sha256']==sha(run/rec['model_file']); continue
   check_time(); start=time.monotonic(); fi=np.flatnonzero(fold!=k); vi=np.flatnonzero(fold==k)
   rec={'name':name,'fold':k,'status':'training','started_utc':now(),'params':params(name),'train_rows':len(fi),'valid_rows':len(vi),'feature_code_sha256':frozen['feature_code_sha256'],'runner_sha256':frozen['runner_sha256']}; dump(result,rec); print('START',name,k,flush=True)
   try:
    X=raw.iloc[fi].reset_index(drop=True); V=raw.iloc[vi].reset_index(drop=True)
    if name=='cb1308':
     cats=X.select_dtypes('object').columns.tolist()
     for c in cats: X[c]=X[c].fillna('__MISSING__').astype(str); V[c]=V[c].fillna('__MISSING__').astype(str)
     model=CatBoostClassifier(**params(name)); model.fit(X,y[fi],cat_features=cats); pred=model.predict_proba(V)[:,1]; mf='model.cbm'; model.save_model(str(run/mf))
    else:
     tr=DistanceFeatures('distance_stats_numeric' if name=='stats639' else 'base'); X=tr.fit_transform_te(X,y[fi]); V=tr.transform_te(V)
     model=lgb.LGBMClassifier(**params(name)); model.fit(X,y[fi],categorical_feature='auto'); pred=model.predict_proba(V)[:,1]; mf='model.txt'; model.booster_.save_model(str(run/mf))
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    np.savez_compressed(run/'validation.npz',idx=vi,id=train.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
    rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),logloss=float(log_loss(y[vi],pred)),model_file=mf,model_sha256=sha(run/mf),elapsed_seconds=time.monotonic()-start,ended_utc=now()); dump(result,rec); print('DONE',name,k,rec['auc'],rec['elapsed_seconds'],flush=True)
    del model,X,V; gc.collect()
   except BaseException as e: rec.update(status='failed',error=repr(e),ended_utc=now()); dump(result,rec); raise
  pred=np.full(len(y),np.nan)
  for k in range(5):
   v=np.load(OUT/f'{name}_fold{k}/validation.npz'); idx=v['idx']; assert np.array_equal(v['target'],y[idx]) and np.array_equal(v['id'],train.id.iloc[idx]); pred[idx]=v['pred']
  assert np.isfinite(pred).all(); np.savez_compressed(OUT/f'{name}_oof.npz',id=train.id.to_numpy(),target=y,fold=fold,pred=pred)
  dump(OUT/f'{name}_summary.json',{'name':name,'auc':float(roc_auc_score(y,pred)),'logloss':float(log_loss(y,pred)),'fold_auc':[float(roc_auc_score(y[fold==k],pred[fold==k])) for k in range(5)],'completed_utc':now()})
 if all((OUT/f'{n}_oof.npz').exists() for n in MODELS):
  ps={n:np.load(OUT/f'{n}_oof.npz')['pred'] for n in MODELS}; old=.4*ps['cb1308']+.4*ps['base639']+.2*ps['bag639']; new=ps['stats639']
  np.savez_compressed(OUT/'submitted_reference_oof.npz',id=train.id.to_numpy(),target=y,fold=fold,pred=old)
  comparison=compare(y,old,new); byfold=[compare(y[fold==k],old[fold==k],new[fold==k]) for k in range(5)]
  result={'pooled':comparison,'by_fold':byfold,'promote_stats':all(x['auc_delta']>0 for x in byfold) and comparison['conditional_95pct_ci'][0]>0,'completed_utc':now()}
  dump(OUT/'comparison.json',result); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()

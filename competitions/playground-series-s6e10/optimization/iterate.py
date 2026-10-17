"""Bounded, official-data-only experiments. Never submits to Kaggle.

Examples: .venv/bin/python -u iterate.py --experiment cb8
          .venv/bin/python -u iterate.py --experiment lgb31 --split confirm --trees 1200
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, platform, time, traceback
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-s6e10-mpl')
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'iteration_artifacts'; OUT.mkdir(exist_ok=True)
CONFIGS={
 'cb8': dict(kind='catboost', iterations=2200, learning_rate=.07, depth=8, l2_leaf_reg=5),
 'cb7': dict(kind='catboost', iterations=2400, learning_rate=.065, depth=7, l2_leaf_reg=5),
 'cb8cats': dict(kind='catboost', iterations=2200, learning_rate=.07, depth=8, l2_leaf_reg=5, feature_mode='rating_categories'),
 'cb6cross': dict(kind='catboost', iterations=2400, learning_rate=.07, depth=6, l2_leaf_reg=5, feature_mode='targeted_crosses'),
 'cbbase': dict(kind='catboost', iterations=956, learning_rate=.1, depth=6, l2_leaf_reg=3),
 'lgb31': dict(kind='lightgbm', iterations=3000, learning_rate=.04, num_leaves=31, max_depth=-1, min_child_samples=100, reg_lambda=5),
 'lgb63': dict(kind='lightgbm', iterations=3000, learning_rate=.04, num_leaves=63, max_depth=-1, min_child_samples=150, reg_lambda=10),
 'lgb63cats': dict(kind='lightgbm', iterations=3500, learning_rate=.04, num_leaves=63, max_depth=-1, min_child_samples=150, reg_lambda=10, feature_mode='targeted_categories'),
 'lgb63allcats': dict(kind='lightgbm', iterations=3500, learning_rate=.04, num_leaves=63, max_depth=-1, min_child_samples=150, reg_lambda=10, feature_mode='rating_categories'),
 'lgb31fe': dict(kind='lightgbm', iterations=3000, learning_rate=.04, num_leaves=31, max_depth=-1, min_child_samples=100, reg_lambda=5, feature_mode='ratings'),
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def features(train,test,kind,mode):
 X=train.drop(columns=['id','satisfaction']).copy(); T=test.drop(columns=['id']).copy()
 cats=X.select_dtypes(include=['object','string']).columns.tolist()
 ratings=[c for c in X.columns if c not in cats and c not in ['Age','Flight Distance','Arrival Delay in Minutes','Departure Delay in Minutes']]
 if mode in ('rating_categories','targeted_categories','targeted_crosses'):
  chosen=ratings if mode=='rating_categories' else ['Inflight wifi service','Online boarding','Ease of Online booking','Gate location']
  for c in chosen:
   X[c+'_cat']=X[c].astype(str); T[c+'_cat']=T[c].astype(str); cats.append(c+'_cat')
  if mode=='targeted_crosses':
   for a,b in [('Inflight wifi service','Type of Travel'),('Online boarding','Class'),('Inflight wifi service','Ease of Online booking')]:
    c=a+'_cross_'+b
    X[c]=X[a].astype(str)+'|'+X[b].astype(str)
    T[c]=T[a].astype(str)+'|'+T[b].astype(str);cats.append(c)
 elif mode=='ratings':
  for frame in [X,T]:
   frame['rating_mean']=frame[ratings].mean(axis=1)
   frame['rating_std']=frame[ratings].std(axis=1)
   frame['rating_zero_count']=(frame[ratings]==0).sum(axis=1)
   frame['rating_low_count']=(frame[ratings]<=2).sum(axis=1)
   frame['rating_high_count']=(frame[ratings]>=4).sum(axis=1)
   frame['delay_difference']=frame['Arrival Delay in Minutes']-frame['Departure Delay in Minutes']
   for a,b in [('Inflight wifi service','Ease of Online booking'),('Online boarding','Inflight wifi service'),('Seat comfort','Inflight entertainment')]:
    frame[a+'_minus_'+b]=frame[a]-frame[b]
 if kind=='catboost':
  for c in cats:
   X[c]=X[c].fillna('__MISSING__').astype(str);T[c]=T[c].fillna('__MISSING__').astype(str)
 else:
  for c in cats:
   # Only train features define category vocabulary. No labels/test-derived statistics.
   dtype=pd.CategoricalDtype(sorted(X[c].fillna('__MISSING__').astype(str).unique()))
   X[c]=X[c].fillna('__MISSING__').astype(str).astype(dtype)
   T[c]=T[c].fillna('__MISSING__').astype(str).astype(dtype)
  # Stable safe names prevent JSON feature-name special character warnings.
  mapping={c:f'f{i:02d}' for i,c in enumerate(X.columns)}
  X=X.rename(columns=mapping); T=T.rename(columns=mapping); cats=[mapping[c] for c in cats]
  X.attrs['source_to_model_feature_names']=mapping
 return X,T,cats

def train_model(config,X,y,fit,valid,cats,trees=None,seed=42):
 cfg=dict(config);kind=cfg.pop('kind');cfg.pop('feature_mode',None)
 n=cfg.pop('iterations'); n=trees or n
 if kind=='catboost':
  params=dict(cfg,iterations=n,loss_function='Logloss',eval_metric='AUC',random_seed=seed,
              task_type='CPU',thread_count=6,allow_writing_files=False,verbose=200)
  model=CatBoostClassifier(**params)
  kwargs=dict(cat_features=cats)
  if valid is not None and trees is None:
   kwargs.update(eval_set=(X.iloc[valid],y.iloc[valid]),early_stopping_rounds=150,use_best_model=True)
  model.fit(X.iloc[fit],y.iloc[fit],**kwargs)
  best=model.tree_count_
  resolved=model.get_all_params()
 else:
  import lightgbm as lgb
  params=dict(cfg,n_estimators=n,objective='binary',n_jobs=6,random_state=seed,
              verbosity=-1,colsample_bytree=1.0,subsample=1.0,max_bin=255,
              deterministic=True,force_col_wise=True)
  model=lgb.LGBMClassifier(**params)
  kwargs=dict(categorical_feature=cats)
  if valid is not None and trees is None:
   kwargs.update(eval_set=[(X.iloc[valid],y.iloc[valid])],eval_metric='auc',callbacks=[lgb.early_stopping(150,first_metric_only=True),lgb.log_evaluation(200)])
  model.fit(X.iloc[fit],y.iloc[fit],**kwargs)
  best=model.best_iteration_ or n
  resolved=model.get_params()
 return model,best,resolved

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',choices=CONFIGS,required=True)
 p.add_argument('--split',choices=['primary','confirm','full'],default='primary')
 p.add_argument('--trees',type=int);p.add_argument('--seed',type=int,default=42)
 args=p.parse_args(); key=f'{args.experiment}_{args.split}_s{args.seed}'
 path=OUT/key;path.mkdir(exist_ok=True)
 started=time.monotonic()
 record=dict(experiment=args.experiment,split=args.split,seed=args.seed,config=CONFIGS[args.experiment],fixed_trees=args.trees,
             started_utc=datetime.now(timezone.utc).isoformat(),status='loading',external_data_used=False)
 dump(path/'result.json',record)
 try:
  train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
  assert train.satisfaction.dtype==bool
  y=train.satisfaction.astype('int8')
  X,T,cats=features(train,test,CONFIGS[args.experiment]['kind'],CONFIGS[args.experiment].get('feature_mode'))
  if args.split=='primary':
   split=np.load(ROOT/'artifacts/split_seed42.npz');fit,valid=split['train'],split['valid']
  elif args.split=='confirm':
   fit,valid=train_test_split(np.arange(len(y)),test_size=.2,random_state=20261017,stratify=y)
   np.savez_compressed(OUT/'split_confirm20261017.npz',train=fit,valid=valid)
  else:
   assert args.trees,'Full refit requires frozen validation-selected tree count'
   fit=np.arange(len(y));valid=None
  record.update(status='training',features=list(X.columns),categorical_features=cats,source_to_model_feature_names=X.attrs.get('source_to_model_feature_names'),train_rows=len(fit),valid_rows=0 if valid is None else len(valid))
  dump(path/'result.json',record)
  model,best,resolved=train_model(CONFIGS[args.experiment],X,y,fit,valid,cats,args.trees,args.seed)
  record.update(best_iterations=int(best),resolved_parameters=resolved,training_seconds=time.monotonic()-started)
  kind=CONFIGS[args.experiment]['kind']
  if kind=='catboost':model.save_model(str(path/'model.cbm'))
  else:model.booster_.save_model(str(path/'model.txt'))
  if valid is not None:
   pred=model.predict_proba(X.iloc[valid])[:,1]
   record.update(validation_auc=float(roc_auc_score(y.iloc[valid],pred)),validation_logloss=float(log_loss(y.iloc[valid],pred)))
   np.savez_compressed(path/'validation.npz',idx=valid,id=train.id.iloc[valid].to_numpy(),target=y.iloc[valid].to_numpy(),pred=pred)
  else:
   pred=model.predict_proba(T)[:,1]
   sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert sample.id.equals(test.id)
   sample['satisfaction']=pred
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
   csv=path/f'submission_{args.experiment}.csv';sample.to_csv(csv,index=False)
   back=pd.read_csv(csv);assert back.id.equals(test.id) and back.shape==sample.shape
   assert np.allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
   record.update(submission=dict(path=str(csv),sha256=sha(csv),rows=len(back),columns=list(back.columns),minimum=float(pred.min()),maximum=float(pred.max()),mean=float(pred.mean()),std=float(pred.std()),readback_verified=True))
  record.update(status='complete',elapsed_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat())
  dump(path/'result.json',record)
  print('RESULT',json.dumps(record),flush=True)
 except BaseException as e:
  record.update(status='failed',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
  dump(path/'result.json',record);raise

if __name__=='__main__':main()

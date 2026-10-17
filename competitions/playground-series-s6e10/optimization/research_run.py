"""Bounded CPU feature ablations, using existing row indices and saved references."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,platform,time,traceback
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-s6e10-mpl')
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss
from research_features import DistanceFeatures
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts';OUT.mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def previous(split):
 ps=[]
 for name,w in [('cb8',.4),('lgb63cats',.4),('lgb63cats_bagging',.2)]:
  p=ROOT/'iteration_artifacts'/f'{name}_{split}_s{20261017 if name.endswith("bagging") else 42}'/'validation.npz'
  v=np.load(p);ps.append((v,w))
 ref=ps[0][0]
 for v,w in ps:
  assert np.array_equal(v['idx'],ref['idx']) and np.array_equal(v['target'],ref['target'])
 return ref,sum(w*v['pred'] for v,w in ps)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['distance_category','distance_frequency','distance_group_stats','distance_te'],required=True);p.add_argument('--split',choices=['primary','confirm'],default='primary');p.add_argument('--trees',type=int,default=639);p.add_argument('--early-stop',action='store_true');p.add_argument('--leaves',type=int,default=63);p.add_argument('--tag',default='a1');p.add_argument('--te-interactions',action='store_true');a=p.parse_args()
 protocol=json.loads((ROOT/'research_protocol.json').read_text());assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
 run=OUT/f'{a.tag}_{a.mode}_{a.split}';run.mkdir(exist_ok=False)
 started=time.monotonic();record={'mode':a.mode,'split':a.split,'status':'loading','started_utc':datetime.now(timezone.utc).isoformat(),'fixed_trees':None if a.early_stop else a.trees,'maximum_trees':a.trees,'early_stopping':a.early_stop,'model_seed':42,'feature_code_sha256':sha(ROOT/'research_features.py'),'runner_sha256':sha(__file__),'cpu_threads':6,'external_data_used':False}
 dump(run/'result.json',record)
 try:
  train=pd.read_csv(ROOT/'data/train.csv');y=train.satisfaction.astype('int8').to_numpy();raw=train.drop(columns=['id','satisfaction'])
  split=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));fi,vi=split['train'],split['valid']
  te=[['Flight Distance']] if a.mode=='distance_te' else []
  if a.te_interactions:te += [['Flight Distance','Class'],['Flight Distance','Type of Travel']]
  transformer=DistanceFeatures(a.mode,te_columns=te)
  X=transformer.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=transformer.transform_te(raw.iloc[vi].reset_index(drop=True))
  assert list(X.columns)==list(V.columns)
  params={'n_estimators':a.trees,'learning_rate':.04,'num_leaves':a.leaves,'max_depth':-1,'min_child_samples':150,'reg_lambda':10,'objective':'binary','n_jobs':6,'random_state':42,'verbosity':-1,'colsample_bytree':1.0,'subsample':1.0,'max_bin':255,'deterministic':True,'force_col_wise':True}
  record.update(status='training',params=params,features=list(X.columns),source_feature_names=transformer.names,categorical_features=list(X.select_dtypes('category').columns),train_rows=len(fi),valid_rows=len(vi),te_audit=transformer.te_audit,preprocessing_seconds=time.monotonic()-started)
  dump(run/'result.json',record)
  model=lgb.LGBMClassifier(**params);kwargs={}
  if a.early_stop:kwargs={'eval_set':[(V,y[vi])],'eval_metric':'auc','callbacks':[lgb.early_stopping(150,first_metric_only=True),lgb.log_evaluation(200)]}
  model.fit(X,y[fi],categorical_feature='auto',**kwargs);pred=model.predict_proba(V)[:,1]
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
  model.booster_.save_model(str(run/'model.txt'));np.savez_compressed(run/'validation.npz',idx=vi,id=train.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
  ref,base=previous(a.split);assert np.array_equal(ref['idx'],vi) and np.array_equal(ref['target'],y[vi])
  component=np.load(ROOT/'iteration_artifacts'/f'lgb63cats_{a.split}_s42/validation.npz')
  record.update(status='complete',best_iterations=int(model.best_iteration_ or model.booster_.num_trees()),actual_trees=model.booster_.num_trees(),validation_auc=float(roc_auc_score(y[vi],pred)),validation_logloss=float(log_loss(y[vi],pred)),vs_existing_lgb63cats=compare(y[vi],component['pred'],pred),vs_submitted_ensemble=compare(y[vi],base,pred),elapsed_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha(run/'model.txt'))
  dump(run/'result.json',record);print(json.dumps(record,indent=2),flush=True)
 except BaseException as e:
  record.update(status='failed',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);dump(run/'result.json',record);raise
if __name__=='__main__':main()

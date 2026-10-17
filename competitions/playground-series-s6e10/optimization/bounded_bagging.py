"""One pre-registered stochastic LightGBM test, with no weight/seed search."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time,traceback
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-s6e10-mpl')
root=Path(__file__).resolve().parent
plan=json.loads((root/'bagging_plan.json').read_text());cutoff=datetime.fromisoformat(plan['training_cutoff_utc'])
def now():return datetime.now(timezone.utc)
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def folder(stage):return root/'iteration_artifacts'/f'lgb63cats_bagging_{stage}_s20261017'
def fit(stage):
 import numpy as np
 import pandas as pd
 import lightgbm as lgb
 from sklearn.metrics import roc_auc_score,log_loss
 from iterate import features
 assert now()<cutoff
 out=folder(stage);out.mkdir(exist_ok=False)
 start=time.monotonic();record={'experiment':plan['experiment'],'split':stage,'seed':20261017,'config':{'kind':'lightgbm','feature_mode':'targeted_categories','parameters':plan['parameters']},'fixed_trees':639,'started_utc':now().isoformat(),'status':'loading','external_data_used':False}
 dump(out/'result.json',record)
 try:
  train=pd.read_csv(root/'data/train.csv');test=pd.read_csv(root/'data/test.csv');assert train.satisfaction.dtype==bool
  y=train.satisfaction.astype('int8');X,T,cats=features(train,test,'lightgbm','targeted_categories')
  if stage=='full':idx=np.arange(len(train));valid=None
  else:
   split=np.load(root/('artifacts/split_seed42.npz' if stage=='primary' else 'iteration_artifacts/split_confirm20261017.npz'))
   idx,valid=split['train'],split['valid']
  record.update(status='training',train_rows=len(idx),valid_rows=0 if valid is None else len(valid),features=list(X.columns),categorical_features=cats)
  dump(out/'result.json',record);model=lgb.LGBMClassifier(**plan['parameters']);model.fit(X.iloc[idx],y.iloc[idx],categorical_feature=cats)
  assert model.booster_.num_trees()==639
  model.booster_.save_model(str(out/'model.txt'));record.update(best_iterations=639,resolved_parameters=model.get_params())
  if valid is not None:
   pred=model.predict_proba(X.iloc[valid])[:,1]
   np.savez_compressed(out/'validation.npz',idx=valid,id=train.id.iloc[valid].to_numpy(),target=y.iloc[valid].to_numpy(),pred=pred)
   record.update(validation_auc=float(roc_auc_score(y.iloc[valid],pred)),validation_logloss=float(log_loss(y.iloc[valid],pred)))
  else:
   pred=model.predict_proba(T)[:,1];sample=pd.read_csv(root/'data/sample_submission.csv');assert sample.id.equals(test.id)
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();sample['satisfaction']=pred
   p=out/'submission_bagged_component.csv';sample.to_csv(p,index=False)
   back=pd.read_csv(p);assert back.id.equals(test.id) and np.allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
   record['submission_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  record.update(status='complete',ended_utc=now().isoformat(),elapsed_seconds=time.monotonic()-start);dump(out/'result.json',record)
 except BaseException as error:
  record.update(status='failed',error=repr(error),traceback=traceback.format_exc());dump(out/'result.json',record);raise

def assess(stage):
 import numpy as np
 from compare_validation import compare
 new=np.load(folder(stage)/'validation.npz');a=np.load(root/'iteration_artifacts'/f'cb8_{stage}_s42/validation.npz');b=np.load(root/'iteration_artifacts'/f'lgb63cats_{stage}_s42/validation.npz')
 for key in ['idx','id','target']:assert np.array_equal(new[key],a[key]) and np.array_equal(new[key],b[key])
 assert np.isfinite(new['pred']).all() and ((new['pred']>=0)&(new['pred']<=1)).all()
 current=.5*a['pred']+.5*b['pred'];candidate=.4*a['pred']+.4*b['pred']+.2*new['pred']
 return compare(new['target'],current,candidate)

def queue():
 assert hashlib.sha256((root/'submission_cb8_lgb63cats_frozen.csv').read_bytes()).hexdigest()==plan['current_best_sha256']
 report={'plan':plan,'started_utc':now().isoformat(),'comparisons':{}}
 for stage in ['primary','confirm']:
  remaining=(cutoff-now()).total_seconds();assert remaining>0
  with (root/f'bagging_{stage}.log').open('x') as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
  report['comparisons'][stage]=assess(stage);dump(root/'bagging_progress.json',report)
  print(stage,json.dumps(report['comparisons'][stage]),flush=True)
 a=report['comparisons']['primary'];b=report['comparisons']['confirm']
 passed=a['delta_auc']>0 and b['delta_auc']>0 and b['conditional_95pct_ci'][0]>0
 report['decision']='eligible_for_full_refit' if passed else 'retain_submitted_best'
 report['ended_utc']=now().isoformat();report['caveat']='Conditional intervals exclude selection uncertainty; confirmation reuses official training samples.'
 dump(root/'bagging_result.json',report);print(json.dumps(report,indent=2),flush=True)
 if passed:
  remaining=(cutoff-now()).total_seconds()
  if remaining<60:report['decision']='retain_best_insufficient_full_fit_and_delivery_time';dump(root/'bagging_result.json',report);return
  with (root/'bagging_full.log').open('x') as log:
   subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--stage','full'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
  print('FULL_READY',flush=True)

p=argparse.ArgumentParser();p.add_argument('--stage',choices=['primary','confirm','full','queue'],default='queue');args=p.parse_args()
if args.stage=='queue':queue()
else:fit(args.stage)

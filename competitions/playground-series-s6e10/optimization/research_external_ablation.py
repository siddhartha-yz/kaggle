"""Small prespecified ablation of independent original-data features on synthetic folds."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,argparse,hashlib,signal
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from research_features import DistanceFeatures
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def external(mode,ids):
 p=ROOT/'research_artifacts/original_teacher_v1';record=json.loads((p/'result.json').read_text());assert record['status']=='complete' and record['model_sha256']==sha(p/'model.txt')
 z=np.load(p/'train_features.npz');assert np.array_equal(ids,z['id']);ext=pd.DataFrame(z['values'],columns=z['columns'])
 cols=[c for c in ext if (c.startswith('orig_teacher') if mode=='teacher' else c.startswith('orig_lookup') if mode=='lookup' else True)]
 return ext[cols],ext.orig_lookup1_logcount.to_numpy()>0
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['teacher','lookup','both'],required=True);p.add_argument('--split',choices=['primary','confirm'],default='primary');a=p.parse_args()
 protocol=json.loads((ROOT/'research_protocol.json').read_text());assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
 run=ROOT/'research_artifacts'/f'e1_original_{a.mode}_{a.split}';run.mkdir(exist_ok=False);start=time.monotonic()
 rec={'mode':a.mode,'split':a.split,'status':'loading','started_utc':datetime.now(timezone.utc).isoformat(),'runner_sha256':sha(__file__),'preprocessing_sha256':sha(ROOT/'research_features.py'),'source_teacher_result_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/result.json'),'fixed_trees':639};dump(run/'result.json',rec)
 try:
  df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,matched=external(a.mode,df.id)
  split=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));fi,vi=split['train'],split['valid']
  tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
  for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
  params=dict(n_estimators=639,learning_rate=.04,num_leaves=63,max_depth=-1,min_child_samples=150,reg_lambda=10,objective='binary',n_jobs=6,random_state=42,verbosity=-1,colsample_bytree=1.,subsample=1.,max_bin=255,deterministic=True,force_col_wise=True)
  rec.update(status='training',params=params,external_features=list(ext),train_rows=len(fi),valid_rows=len(vi));dump(run/'result.json',rec)
  model=lgb.LGBMClassifier(**params);model.fit(X,y[fi]);pred=model.predict_proba(V)[:,1];assert np.isfinite(pred).all();model.booster_.save_model(str(run/'model.txt'));np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
  baseline=np.load(ROOT/'research_artifacts'/f'b1_distance_stats_numeric_{a.split}'/'validation.npz');assert np.array_equal(baseline['idx'],vi) and np.array_equal(baseline['target'],y[vi]);base=baseline['pred']
  slices={}
  for name,m in [('near_match',matched[vi]),('no_near_match',~matched[vi])]:
   slices[name]={'rows':int(m.sum()),'positive_rate':float(y[vi][m].mean()),'comparison':compare(y[vi][m],base[m],pred[m])}
  rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),vs_stats=compare(y[vi],base,pred),near_key_slices=slices,model_sha256=sha(run/'model.txt'),elapsed_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat());dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e));dump(run/'result.json',rec);raise
if __name__=='__main__':main()

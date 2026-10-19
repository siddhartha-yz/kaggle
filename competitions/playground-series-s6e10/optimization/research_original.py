"""Public CC0 original-data teacher and lookup features; no synthetic targets used."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,signal,argparse
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,log_loss
from research_features import DistanceFeatures
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts'/'original_teacher_v1'
NEAR=['Flight Distance','Age','Gender','Class','Customer Type','Type of Travel']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def normalized_hash(df,cols):
 x=df[cols].copy()
 for c in x:
  if pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].astype('float64')
  else:x[c]=x[c].fillna('__MISSING__').astype(str)
 return pd.util.hash_pandas_object(x,index=False).to_numpy()
def safe_original():
 source=ROOT/'research_sources/original_cc0/data.csv';assert sha(source)=='2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3'
 orig=pd.read_csv(source);train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
 cols=list(train.drop(columns=['id','satisfaction']).columns);assert set(orig.columns)==set(cols+['satisfaction']);assert orig.satisfaction.dtype==bool
 h=normalized_hash(orig,cols);hs=np.concatenate([normalized_hash(train,cols),normalized_hash(test,cols)])
 keep=~np.isin(h,hs);saved=np.load(ROOT/'research_sources/original_safe_mask.npy');assert np.array_equal(keep,saved) and keep.sum()==129859
 safe=orig.loc[keep].reset_index(drop=True)
 g=safe.groupby(NEAR,dropna=False).satisfaction.agg(['count','sum','nunique'])
 audit={'source_sha256':sha(source),'safe_mask_sha256':sha(ROOT/'research_sources/original_safe_mask.npy'),'safe_rows':len(safe),'excluded_exact_rows':int((~keep).sum()),'near_key_groups':len(g),'near_key_groups_with_conflicting_original_labels':int((g['nunique']>1).sum()),'original_rows_in_conflicting_near_groups':int(g.loc[g['nunique']>1,'count'].sum()),'target_provenance':'Only CC0 original labels. No synthetic labels used to fit teacher or lookup. Exact matches to any synthetic train/test vector excluded.'}
 if (ROOT/'original_exact_label_audit.json').exists():audit['exact_label_audit']=json.loads((ROOT/'original_exact_label_audit.json').read_text())
 return safe,train,test,cols,audit
class OriginalLookup:
 def fit(self,X,y):
  self.maps=[];self.prior=float(np.mean(y));self.specs=[(['Flight Distance'],100.),(NEAR,20.)]
  for cols,alpha in self.specs:
   keys=DistanceFeatures._key(X,cols);d=pd.DataFrame({'key':keys,'y':np.asarray(y)});g=d.groupby('key').y.agg(['sum','count']);self.maps.append(((g['sum']+alpha*self.prior)/(g['count']+alpha),g['count']))
  return self
 def transform(self,X):
  out={}
  for i,((cols,alpha),(means,counts)) in enumerate(zip(self.specs,self.maps)):
   key=DistanceFeatures._key(X,cols);out[f'orig_lookup{i}_mean']=key.map(means).fillna(self.prior).to_numpy();out[f'orig_lookup{i}_logcount']=np.log1p(key.map(counts).fillna(0).to_numpy())
  return pd.DataFrame(out)
def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();p=json.loads((ROOT/'research_protocol.json').read_text());assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(p['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
 original,train,test,cols,audit=safe_original();dump(OUT/'audit.json',audit)
 y=original.satisfaction.astype('int8').to_numpy();raw=original[cols]
 fi,vi=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=20261018)
 np.savez_compressed(OUT/'original_selection_split.npz',train=fi,valid=vi)
 params=dict(n_estimators=3000,learning_rate=.04,num_leaves=63,min_child_samples=100,reg_lambda=10,objective='binary',n_jobs=6,random_state=20261018,verbosity=-1,deterministic=True,force_col_wise=True)
 plan={'params':params,'early_stopping_original_only':150,'external_source':audit,'synthetic_target_used':False,'feature_code_sha256':sha(ROOT/'research_features.py'),'runner_sha256':sha(__file__),'frozen_utc':datetime.now(timezone.utc).isoformat()};dump(OUT/'plan.json',plan)
 tr=DistanceFeatures('base');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
 model=lgb.LGBMClassifier(**params);model.fit(X,y[fi],eval_set=[(V,y[vi])],eval_metric='auc',callbacks=[lgb.early_stopping(150,first_metric_only=True),lgb.log_evaluation(200)])
 pred=model.predict_proba(V)[:,1];best=int(model.best_iteration_);selection={'auc':float(roc_auc_score(y[vi],pred)),'logloss':float(log_loss(y[vi],pred)),'best_trees':best};dump(OUT/'original_selection_result.json',selection);model.booster_.save_model(str(OUT/'selection_model.txt'))
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 params['n_estimators']=best;full=DistanceFeatures('base');X=full.fit_transform_te(raw,y);model=lgb.LGBMClassifier(**params);model.fit(X,y);model.booster_.save_model(str(OUT/'model.txt'))
 lookup=OriginalLookup().fit(raw,y)
 for name,df in [('train',train),('test',test)]:
  raw_x=df[cols].reset_index(drop=True);features=lookup.transform(raw_x);features['orig_teacher_lgb']=model.predict_proba(full.transform_te(raw_x))[:,1];assert np.isfinite(features.to_numpy()).all()
  np.savez_compressed(OUT/f'{name}_features.npz',id=df.id.to_numpy(),values=features.to_numpy(),columns=np.asarray(features.columns,dtype=str))
  audit[f'{name}_near_key_matched_rows']=int((features.orig_lookup1_logcount>0).sum())
 audit.update(status='complete',selection=selection,full_trees=best,model_sha256=sha(OUT/'model.txt'),elapsed_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat());dump(OUT/'result.json',audit);print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()

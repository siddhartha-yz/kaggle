"""Nested holdout meta training; no outer validation label enters any fitted object."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,signal,argparse,gc
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb,joblib
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from research_features import DistanceFeatures
from research_external_ablation import external
from research_oof import params
from compare_validation import compare
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc)
def logits(pred,teacher,extra=False):
 z=np.column_stack([logit(np.clip(pred,1e-6,1-1e-6)),logit(np.clip(teacher,1e-6,1-1e-6))])
 if extra:z=np.column_stack([z,z[:,1]-z[:,0],z[:,2]-z[:,0],z[:,3]-z[:,0]])
 return z

def outer_predictions(split,vi,df,fold=None):
 if split=='oof':paths=['lgb_stats_orig_both_oof/oof.npz','cb_stats_orig_both_oof/oof.npz','residual_prior05_oof_t581/oof.npz']
 else:paths=[f'e1_original_both_{split}/validation.npz',f'cb_stats_orig_both_{split}/validation.npz',('residual_prior05_primary_t2200_es' if split=='primary' else 'residual_prior05_confirm_t581')+'/validation.npz']
 predictions=[]
 for path in paths:
  z=np.load(ROOT/'research_artifacts'/path)
  if split=='oof':
   assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],df.satisfaction.astype('int8')) and np.array_equal(z['fold'],fold);pred=z['pred'][vi]
  else:
   assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],df.satisfaction.iloc[vi].astype('int8'));pred=z['pred']
  predictions.append(pred)
 return np.column_stack(predictions)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['primary','confirm','oof'],required=True);parser.add_argument('--method',choices=['both','logistic','residual_tree'],default='both');a=parser.parse_args();assert a.split=='primary' or a.method!='both'
 assert json.loads((ROOT/'research_stage8_queue_state.json').read_text())['status']=='complete'
 protocol=json.loads((ROOT/'research_protocol.json').read_text());frozen=json.loads((ROOT/'research_nested_stack_plan.json').read_text())
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-now()).total_seconds())))
 methods=['logistic','residual_tree'] if a.method=='both' else [a.method];out=ROOT/'research_artifacts'/f'nested_stack_{a.split}';out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();fold=None
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)]
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])]
 plan={'split':a.split,'methods':methods,'frozen_plan_sha256':sha(ROOT/'research_nested_stack_plan.json'),'runner_sha256':sha(__file__),'preprocessing_sha256':sha(ROOT/'research_features.py'),'source_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'frozen_utc':now().isoformat()}
 if (out/'plan.json').exists():
  old=json.loads((out/'plan.json').read_text());assert all(old[k]==plan[k] for k in ['split','methods','frozen_plan_sha256','preprocessing_sha256','source_features_sha256'])
 else:dump(out/'plan.json',plan)
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc']);start=time.monotonic()
  inner_train,inner_meta=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020)
  assert not np.intersect1d(inner_train,inner_meta).size and not np.intersect1d(fi,vi).size
  assert np.array_equal(np.sort(np.concatenate([inner_train,inner_meta])),np.sort(fi))
  np.savez_compressed(run/'nested_indices.npz',outer_train=fi,outer_valid=vi,inner_train=inner_train,inner_meta=inner_meta)
  rec={'status':'preprocessing','fold':k,'outer_train_rows':len(fi),'outer_valid_rows':len(vi),'inner_train_rows':len(inner_train),'inner_meta_rows':len(inner_meta),'nested_indices_sha256':sha(run/'nested_indices.npz'),'outer_valid_excluded_from_all_fitting':True,'started_utc':now().isoformat()};dump(run/'result.json',rec)
  try:
   tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[inner_train].reset_index(drop=True),y[inner_train]);V=tr.transform_te(raw.iloc[inner_meta].reset_index(drop=True))
   for c in ext:X[c]=ext[c].to_numpy()[inner_train];V[c]=ext[c].to_numpy()[inner_meta]
   joblib.dump(tr,run/'inner_transformer.joblib');inner_predictions=[];model_hashes={}
   for family in ['lgb_original','cb_original','residual_prior05']:
    assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc']);rec['status']='training_'+family;dump(run/'result.json',rec);print('START',a.split,k,family,now().isoformat(),flush=True)
    cached=run/(family+'_inner_prediction.npz')
    if cached.exists():
     z=np.load(cached);assert np.array_equal(z['idx'],inner_meta) and np.array_equal(z['target'],y[inner_meta]);pred=z['pred'];mf=run/(family+('.cbm' if family=='cb_original' else '.txt'));assert mf.exists()
    elif family=='cb_original':
     CX=X.copy();CV=V.copy();cats=CX.select_dtypes('category').columns.tolist()
     for c in cats:CX[c]=CX[c].astype(str);CV[c]=CV[c].astype(str)
     model=CatBoostClassifier(**params('cb1308'));model.fit(CX,y[inner_train],cat_features=cats);pred=model.predict_proba(CV)[:,1];mf=run/(family+'.cbm');model.save_model(str(mf));del CX,CV,model
    else:
     config=params('stats639');kwargs={}
     if family=='residual_prior05':config['n_estimators']=581;kwargs['init_score']=.5*logit(np.clip(teacher[inner_train],1e-6,1-1e-6))
     model=lgb.LGBMClassifier(**config);model.fit(X,y[inner_train],**kwargs);mf=run/(family+'.txt');model.booster_.save_model(str(mf))
     pred=expit(model.booster_.predict(V,raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[inner_meta],1e-6,1-1e-6))) if family=='residual_prior05' else model.predict_proba(V)[:,1];del model
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    if not cached.exists():np.savez_compressed(cached,idx=inner_meta,target=y[inner_meta],pred=pred)
    model_hashes[family]=sha(mf);inner_predictions.append(pred);print('DONE',a.split,k,family,flush=True)
   P=np.column_stack(inner_predictions);O=outer_predictions(a.split,vi,df,fold);rec['model_hashes']=model_hashes;rec['methods']={}
   for method in methods:
    rec['status']='meta_'+method;dump(run/'result.json',rec);extra=method=='residual_tree';M=logits(P,teacher[inner_meta],extra);Q=logits(O,teacher[vi],extra)
    if method=='logistic':
     model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,solver='lbfgs'));model.fit(M,y[inner_meta]);pred=model.predict_proba(Q)[:,1];joblib.dump(model,run/'meta_logistic.joblib');loaded=joblib.load(run/'meta_logistic.joblib');again=loaded.predict_proba(Q)[:,1];details={'standardized_coefficients':model[-1].coef_.tolist(),'intercept':model[-1].intercept_.tolist()}
    else:
     config=dict(n_estimators=200,learning_rate=.02,num_leaves=4,max_depth=2,min_child_samples=2000,reg_lambda=100,objective='binary',n_jobs=6,random_state=42,verbosity=-1,colsample_bytree=1.,subsample=1.,max_bin=255,deterministic=True,force_col_wise=True)
     model=lgb.LGBMClassifier(**config);model.fit(M,y[inner_meta],init_score=M[:,0]);pred=expit(model.booster_.predict(Q,raw_score=True,num_threads=6)+Q[:,0]);model.booster_.save_model(str(run/'meta_residual_tree.txt'));loaded=lgb.Booster(model_file=str(run/'meta_residual_tree.txt'));again=expit(loaded.predict(Q,raw_score=True,num_threads=6)+Q[:,0]);details={'params':config}
    assert np.isfinite(pred).all() and np.max(np.abs(pred-again))<1e-14
    np.savez_compressed(run/f'{method}_validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
    rec['methods'][method]={'comparison':compare(y[vi],O[:,0],pred),'reload_maxdiff':float(np.max(np.abs(pred-again))),**details};del model,loaded
   rec.update(status='complete',elapsed_seconds=time.monotonic()-start,ended_utc=now().isoformat());dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del X,V,P,O;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now().isoformat());dump(run/'result.json',rec);raise
 result={'status':'complete','split':a.split,'methods':{},'completed_utc':now().isoformat()}
 for method in methods:
  if a.split=='oof':
   pred=np.full(len(y),np.nan)
   for k,(fi,vi) in enumerate(splits):z=np.load(out/f'fold{k}/{method}_validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
   assert np.isfinite(pred).all();target=y;ids=df.id.to_numpy();base=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz')['pred'];np.savez_compressed(out/f'{method}_oof.npz',id=ids,target=target,fold=fold,pred=pred);byfold=[compare(y[fold==k],base[fold==k],pred[fold==k]) for k in range(5)]
  else:
   vi=splits[0][1];z=np.load(out/f'fold0/{method}_validation.npz');pred=z['pred'];target=y[vi];ids=df.id.iloc[vi].to_numpy();base=outer_predictions(a.split,vi,df)[:,0];byfold=[]
  c=compare(target,base,pred);result['methods'][method]={'comparison':c,'byfold':byfold}
  if a.split=='oof':result['methods'][method]['eligible']=c['delta_auc']>=.0002 and c['conditional_95pct_ci'][0]>0 and all(v['delta_auc']>0 for v in byfold)
 dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

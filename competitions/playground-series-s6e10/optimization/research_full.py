"""Frozen promoted full-data refit, reproducible saved transforms and strict CSV audit."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,signal,argparse
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb,joblib
from research_features import DistanceFeatures
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',choices=['stats639','stats_original639'],default='stats639');a=p.parse_args();uses_original=a.name=='stats_original639'
 protocol=json.loads((ROOT/'research_protocol.json').read_text());assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
 comparison=json.loads((ROOT/('research_artifacts/lgb_stats_orig_both_oof/result.json' if uses_original else 'research_artifacts/oof_v1/comparison.json')).read_text());assert comparison['stable_positive_vs_stats' if uses_original else 'promote_stats'] is True
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
 out=ROOT/'research_artifacts'/f'{a.name}_full_v1';out.mkdir(exist_ok=False);start=time.monotonic()
 expected={'train.csv':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test.csv':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission.csv':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
 assert all(sha(ROOT/'data'/n)==h for n,h in expected.items())
 train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert sample.id.equals(test.id) and list(sample.columns)==['id','satisfaction'] and len(test)==299844
 y=train.satisfaction.astype('int8').to_numpy();raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest)
 params=json.loads((ROOT/'research_artifacts/oof_v1/plan.json').read_text())['models']['stats639']
 rec={'status':'training','model':a.name,'params':params,'positive_label':True,'external_data_used':uses_original,'train_rows':len(train),'test_rows':len(test),'data_sha256':expected,'oof_comparison':comparison,'runner_sha256':sha(__file__),'feature_code_sha256':sha(ROOT/'research_features.py'),'started_utc':datetime.now(timezone.utc).isoformat()};dump(out/'result.json',rec)
 try:
  tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw,y);T=tr.transform_te(rawtest);joblib.dump(tr,out/'transformer.joblib');rec['source_to_model_feature_names']=tr.names;rec['categorical_features']=list(X.select_dtypes('category').columns)
  if uses_original:
   from research_original import safe_original,OriginalLookup
   source=ROOT/'research_artifacts/original_teacher_v1';orig_record=json.loads((source/'result.json').read_text());assert orig_record['status']=='complete' and sha(source/'model.txt')==orig_record['model_sha256']
   original,_,_,cols,source_audit=safe_original();oy=original.satisfaction.astype('int8').to_numpy();ot=DistanceFeatures('base');ot.fit_transform_te(original[cols],oy);lookup=OriginalLookup().fit(original[cols],oy)
   original_model=lgb.Booster(model_file=str(source/'model.txt'));fresh=lookup.transform(rawtest);fresh['orig_teacher_lgb']=original_model.predict(ot.transform_te(rawtest),num_threads=6)
   for label,frame,ids in [('train',X,train.id),('test',T,test.id)]:
    z=np.load(source/f'{label}_features.npz');assert np.array_equal(z['id'],ids)
    if label=='test':assert list(z['columns'])==list(fresh.columns) and np.allclose(z['values'],fresh.values,rtol=0,atol=1e-14)
    for j,c in enumerate(z['columns']):frame[c]=z['values'][:,j]
   joblib.dump({'teacher_transformer':ot,'lookup':lookup},out/'original_transformers.joblib');rec['original_source_audit']=source_audit;rec['original_model_sha256']=orig_record['model_sha256'];rec['external_feature_file_sha256']={label:sha(source/f'{label}_features.npz') for label in ['train','test']};rec['external_features']=list(fresh.columns)
  model=lgb.LGBMClassifier(**params);model.fit(X,y);assert model.booster_.num_trees()==639;model.booster_.save_model(str(out/'model.txt'));pred=model.predict_proba(T)[:,1]
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all() and pred.std()>0
  saved=lgb.Booster(model_file=str(out/'model.txt'));restored=joblib.load(out/'transformer.joblib');reloadT=restored.transform_te(rawtest)
  if uses_original:
   transforms=joblib.load(out/'original_transformers.joblib');ext=transforms['lookup'].transform(rawtest);ext['orig_teacher_lgb']=original_model.predict(transforms['teacher_transformer'].transform_te(rawtest),num_threads=6)
   for c in ext:reloadT[c]=ext[c].to_numpy()
  again=saved.predict(reloadT,num_threads=6);maxdiff=float(np.max(np.abs(pred-again)));assert maxdiff<1e-14
  stem='submission_distance_original_v4' if uses_original else 'submission_distance_stats_v3';sample.satisfaction=pred;csv=ROOT/(stem+'.csv');assert not csv.exists();sample.to_csv(csv,index=False);back=pd.read_csv(csv)
  assert list(back.columns)==['id','satisfaction'] and len(back)==299844 and back.id.equals(test.id) and back.id.is_unique and back.satisfaction.notna().all() and back.satisfaction.between(0,1).all()
  assert np.allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
  rec.update(status='complete',model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),prediction_reload_maxdiff=maxdiff,elapsed_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat(),submission={'path':str(csv),'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(back),'columns':list(back.columns),'id_order_matches_official_test':True,'min':float(pred.min()),'max':float(pred.max()),'mean':float(pred.mean()),'std':float(pred.std()),'csv_readback_verified':True})
  if uses_original:rec['original_transformers_sha256']=sha(out/'original_transformers.joblib')
  dump(out/'result.json',rec);dump(ROOT/(stem+'_audit.json'),rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e));dump(out/'result.json',rec);raise
if __name__=='__main__':main()

"""Refit only the frozen Stage8 winner after its OOF submission gate passes."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, time, signal
os.environ['OMP_NUM_THREADS']='6'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from research_features import DistanceFeatures
from research_original import safe_original, OriginalLookup
from compare_validation import compare
ROOT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now(): return datetime.now(timezone.utc)

def main():
 protocol=json.loads((ROOT/'research_protocol.json').read_text())
 for stage in [7,8]: assert json.loads((ROOT/f'research_stage{stage}_queue_state.json').read_text())['status']=='complete', 'Never overlap the serial validation queue'
 selection=json.loads((ROOT/'research_stage8_selection.json').read_text()); chosen=selection['chosen']; assert chosen is not None
 assert selection['confirmation']['conditional_95pct_ci'][0]>0 and 'oof_result_path' in selection
 resultpath=Path(selection['oof_result_path']); result=json.loads(resultpath.read_text()); plan=json.loads((resultpath.parent/'plan.json').read_text())
 assert result['promote_alternative'] is True
 base=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz'); candidate=np.load(resultpath.parent/'oof.npz')
 for key in ['id','target','fold']: assert np.array_equal(base[key],candidate[key])
 comparison=compare(base['target'],base['pred'],candidate['pred'])
 byfold=[compare(base['target'][base['fold']==k],base['pred'][base['fold']==k],candidate['pred'][base['fold']==k]) for k in range(5)]
 assert comparison['delta_auc']>=.0002 and comparison['conditional_95pct_ci'][0]>0 and all(c['delta_auc']>0 for c in byfold)
 assert plan['preprocessing_sha256']==sha(ROOT/'research_features.py')
 assert plan['external_sha256']==sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')
 if chosen['representation']=='distribution': assert plan['distribution_preprocessing_sha256']==sha(ROOT/'research_distributions.py')
 if chosen['external']=='both_xgb': assert plan['xgb_teacher_sha256']==sha(ROOT/'research_artifacts/original_xgb_v1/train_features.npz')
 assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
 def stop(s,f): raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop); signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-now()).total_seconds())))
 expected={'train.csv':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test.csv':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission.csv':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
 assert all(sha(ROOT/'data'/n)==h for n,h in expected.items())
 train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
 assert len(test)==299844 and sample.id.equals(test.id) and list(sample)==['id','satisfaction']
 y=train.satisfaction.astype('int8').to_numpy();raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id')
 assert list(raw)==list(rawtest) and np.array_equal(candidate['id'],train.id) and np.array_equal(candidate['target'],y)
 out=ROOT/'research_artifacts'/('stage8_'+chosen['name']+'_full_v1'); out.mkdir(exist_ok=False);start=time.monotonic()
 rec={'status':'preparing','chosen':chosen,'params':plan['params'],'source_oof_plan':plan,'comparison':comparison,'byfold':byfold,'data_sha256':expected,'positive_label':True,'runner_sha256':sha(__file__),'started_utc':now().isoformat()};dump(out/'result.json',rec)
 try:
  if chosen['representation']=='distribution':
   from research_distributions import DistanceDistributions
   tr=DistanceDistributions()
  elif chosen['representation']=='interactions': tr=DistanceFeatures('distance_stats_numeric',te_columns=plan['inner_oof_interactions']['keys'],alpha=20.)
  else: tr=DistanceFeatures('distance_stats_numeric')
  X=tr.fit_transform_te(raw,y);T=tr.transform_te(rawtest);joblib.dump(tr,out/'transformer.joblib');rec['te_audit']=tr.te_audit
  original,_,_,cols,source_audit=safe_original();oy=original.satisfaction.astype('int8').to_numpy();ot=DistanceFeatures('base');ot.fit_transform_te(original[cols],oy);lookup=OriginalLookup().fit(original[cols],oy)
  source=ROOT/'research_artifacts/original_teacher_v1';teacher_record=json.loads((source/'result.json').read_text()); assert sha(source/'model.txt')==teacher_record['model_sha256']
  teacher=lgb.Booster(model_file=str(source/'model.txt'))
  fresh=lookup.transform(rawtest);fresh['orig_teacher_lgb']=teacher.predict(ot.transform_te(rawtest),num_threads=6)
  feature_sources=[source];rec['source_models']={'lgb':teacher_record['model_sha256']}
  if chosen['external']=='both_xgb':
   from xgboost import XGBClassifier
   from research_original_xgb import matrix
   xsource=ROOT/'research_artifacts/original_xgb_v1';xr=json.loads((xsource/'result.json').read_text());assert sha(xsource/'model.json')==xr['model_sha256'] and sha(xsource/'vocabulary.json')==xr['vocabulary_sha256']
   xm=XGBClassifier(n_jobs=6);xm.load_model(xsource/'model.json');xv=json.loads((xsource/'vocabulary.json').read_text());M,_=matrix(rawtest,xv);fresh['orig_teacher_xgb']=xm.predict_proba(M)[:,1];feature_sources.append(xsource);rec['source_models']['xgb']=xr['model_sha256']
  for label,frame,ids in [('train',X,train.id),('test',T,test.id)]:
   for s in feature_sources:
    z=np.load(s/f'{label}_features.npz'); assert np.array_equal(z['id'],ids)
    for j,c in enumerate(z['columns']):
     if label=='test':np.testing.assert_allclose(z['values'][:,j],fresh[c],rtol=0,atol=1e-14)
     frame[c]=z['values'][:,j]
  joblib.dump({'teacher_transformer':ot,'lookup':lookup},out/'original_transformers.joblib');rec['original_source_audit']=source_audit;rec['external_feature_hashes']={str(s.name):{label:sha(s/f'{label}_features.npz') for label in ['train','test']} for s in feature_sources}
  rec['status']='training';dump(out/'result.json',rec)
  model=lgb.LGBMClassifier(**plan['params']);model.fit(X,y);assert model.booster_.num_trees()==plan['params']['n_estimators'];model.booster_.save_model(str(out/'model.txt'));pred=model.predict_proba(T)[:,1]
  restored=joblib.load(out/'transformer.joblib');reloadT=restored.transform_te(rawtest)
  for c in fresh: reloadT[c]=fresh[c].to_numpy()
  saved=lgb.Booster(model_file=str(out/'model.txt'));again=saved.predict(reloadT,num_threads=6);maxdiff=float(np.max(np.abs(pred-again)));assert maxdiff<1e-14
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all() and pred.std()>0
  stem='submission_contrast_v5';csv=ROOT/(stem+'.csv');assert not csv.exists();sample.satisfaction=pred;sample.to_csv(csv,index=False);back=pd.read_csv(csv)
  assert list(back)==['id','satisfaction'] and len(back)==299844 and back.id.equals(test.id) and back.id.is_unique and back.satisfaction.notna().all() and back.satisfaction.between(0,1).all()
  np.testing.assert_allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
  rec.update(status='complete',model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),prediction_reload_maxdiff=maxdiff,elapsed_seconds=time.monotonic()-start,ended_utc=now().isoformat(),submission={'path':str(csv),'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(back),'columns':list(back),'id_order_matches_official_test':True,'min':float(pred.min()),'max':float(pred.max()),'mean':float(pred.mean()),'std':float(pred.std()),'csv_readback_verified':True})
  dump(out/'result.json',rec);dump(ROOT/(stem+'_audit.json'),rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e: rec.update(status='failed',error=repr(e));dump(out/'result.json',rec);raise
if __name__=='__main__':main()

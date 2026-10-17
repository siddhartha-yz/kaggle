"""Gated nested-stack full refit: reuse v4, refit two complements, meta-fit on OOF."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,signal
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,lightgbm as lgb,joblib
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from research_nested_stack import logits
from research_oof import params
from research_original import safe_original
from compare_validation import compare
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc)

def main():
 assert json.loads((ROOT/'research_stage9_queue_state.json').read_text())['status']=='complete'
 selection=json.loads((ROOT/'research_nested_stack_selection.json').read_text());method=selection['chosen'];assert method=='residual_tree' and selection['confirmation']['comparison']['conditional_95pct_ci'][0]>0 and selection['oof']['eligible']
 protocol=json.loads((ROOT/'research_protocol.json').read_text());assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(protocol['training_hard_cutoff_utc'])-now()).total_seconds())))
 out=ROOT/'research_artifacts/nested_stack_full_v1';out.mkdir(exist_ok=False);start=time.monotonic();rec={'status':'preflight','method':method,'selection':selection,'runner_sha256':sha(__file__),'started_utc':now().isoformat()};dump(out/'result.json',rec)
 try:
  reference=ROOT/'research_artifacts/stats_original639_full_v1';rr=json.loads((reference/'result.json').read_text());assert rr['status']=='complete'
  assert all(sha(ROOT/'data'/n)==v for n,v in rr['data_sha256'].items());assert sha(reference/'model.txt')==rr['model_sha256'] and sha(reference/'transformer.joblib')==rr['transformer_sha256'] and sha(reference/'original_transformers.joblib')==rr['original_transformers_sha256']
  assert sha(ROOT/'research_features.py')==rr['feature_code_sha256']
  train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert sample.id.equals(test.id) and list(sample)==['id','satisfaction'] and len(test)==299844
  y=train.satisfaction.astype('int8').to_numpy();raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest)
  folds=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(folds['id'],train.id) and np.array_equal(folds['target'],y)
  oof=[];oof_paths=['lgb_stats_orig_both_oof','cb_stats_orig_both_oof','residual_prior05_oof_t581']
  for name in oof_paths:
   z=np.load(ROOT/'research_artifacts'/name/'oof.npz');assert np.array_equal(z['id'],train.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],folds['fold']);oof.append(z['pred'])
  P=np.column_stack(oof);candidate=np.load(ROOT/'research_artifacts/nested_stack_oof/residual_tree_oof.npz')
  assert np.array_equal(candidate['id'],train.id) and np.array_equal(candidate['target'],y) and np.array_equal(candidate['fold'],folds['fold'])
  c=compare(y,P[:,0],candidate['pred']);bf=[compare(y[folds['fold']==k],P[folds['fold']==k,0],candidate['pred'][folds['fold']==k]) for k in range(5)];assert c['delta_auc']>=.0002 and c['conditional_95pct_ci'][0]>0 and all(v['delta_auc']>0 for v in bf)
  for k in range(5):
   z=np.load(ROOT/f'research_artifacts/nested_stack_oof/fold{k}/nested_indices.npz');vi=np.flatnonzero(folds['fold']==k);assert np.array_equal(z['outer_valid'],vi) and not np.intersect1d(z['inner_train'],vi).size and not np.intersect1d(z['inner_meta'],vi).size
  tr=joblib.load(reference/'transformer.joblib');X=tr.transform_te(raw);T=tr.transform_te(rawtest)
  source=ROOT/'research_artifacts/original_teacher_v1';sr=json.loads((source/'result.json').read_text());assert sha(source/'model.txt')==sr['model_sha256'];teacher_model=lgb.Booster(model_file=str(source/'model.txt'));original_transformers=joblib.load(reference/'original_transformers.joblib')
  original,_,_,_,source_audit=safe_original();del original
  fresh=original_transformers['lookup'].transform(rawtest);fresh['orig_teacher_lgb']=teacher_model.predict(original_transformers['teacher_transformer'].transform_te(rawtest),num_threads=6)
  teachers={}
  for label,frame,ids in [('train',X,train.id),('test',T,test.id)]:
   z=np.load(source/f'{label}_features.npz');assert np.array_equal(z['id'],ids)
   if label=='test':assert list(z['columns'])==list(fresh);np.testing.assert_allclose(z['values'],fresh.values,rtol=0,atol=1e-14)
   for j,name in enumerate(z['columns']):frame[name]=z['values'][:,j]
   teachers[label]=frame.orig_teacher_lgb.to_numpy()
  rec.update(status='training',comparison=c,byfold=bf,data_sha256=rr['data_sha256'],original_source_audit=source_audit,reference_model_sha256=rr['model_sha256'],source_teacher_sha256=sr['model_sha256'],positive_label=True,final_meta_training='All training rows with previously completed genuine OOF predictions from three base families; chosen shallow residual algorithm unchanged.');dump(out/'result.json',rec)
  model=lgb.Booster(model_file=str(reference/'model.txt'));base=model.predict(T,num_threads=6);previous=pd.read_csv(ROOT/'submission_distance_original_v4.csv');assert np.array_equal(previous.id,test.id);np.testing.assert_allclose(base,previous.satisfaction,rtol=0,atol=1e-14);preds=[base];rec['components']={'lgb_original':{'reused':True,'sha256':rr['model_sha256']}}
  for family in ['cb_original','residual_prior05']:
   assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc']);print('START',family,now().isoformat(),flush=True)
   if family=='cb_original':
    CX=X.copy();CT=T.copy();cats=CX.select_dtypes('category').columns.tolist()
    for name in cats:CX[name]=CX[name].astype(str);CT[name]=CT[name].astype(str)
    config=params('cb1308');model=CatBoostClassifier(**config);model.fit(CX,y,cat_features=cats);pred=model.predict_proba(CT,thread_count=6)[:,1];mf=out/'cb_original.cbm';model.save_model(str(mf));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(mf));again=loaded.predict_proba(CT,thread_count=6)[:,1];del CX,CT
   else:
    config=params('stats639');config['n_estimators']=581;prior=.5*logit(np.clip(teachers['train'],1e-6,1-1e-6));testprior=.5*logit(np.clip(teachers['test'],1e-6,1-1e-6));model=lgb.LGBMClassifier(**config);model.fit(X,y,init_score=prior);mf=out/'residual_prior05.txt';model.booster_.save_model(str(mf));pred=expit(model.booster_.predict(T,raw_score=True,num_threads=6)+testprior);loaded=lgb.Booster(model_file=str(mf));again=expit(loaded.predict(T,raw_score=True,num_threads=6)+testprior)
   assert np.isfinite(pred).all() and np.max(np.abs(pred-again))<1e-14;preds.append(pred);rec['components'][family]={'params':config,'model_sha256':sha(mf),'reload_maxdiff':float(np.max(np.abs(pred-again)))};dump(out/'result.json',rec);del model,loaded;print('DONE',family,flush=True)
  O=np.column_stack(preds);M=logits(P,teachers['train'],True);Q=logits(O,teachers['test'],True);meta_params=json.loads((ROOT/'research_artifacts/nested_stack_oof/fold0/result.json').read_text())['methods']['residual_tree']['params']
  for k in range(1,5):assert json.loads((ROOT/f'research_artifacts/nested_stack_oof/fold{k}/result.json').read_text())['methods']['residual_tree']['params']==meta_params
  assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc']);model=lgb.LGBMClassifier(**meta_params);model.fit(M,y,init_score=M[:,0]);pred=expit(model.booster_.predict(Q,raw_score=True,num_threads=6)+Q[:,0]);model.booster_.save_model(str(out/'meta_residual_tree.txt'));loaded=lgb.Booster(model_file=str(out/'meta_residual_tree.txt'));again=expit(loaded.predict(Q,raw_score=True,num_threads=6)+Q[:,0]);assert np.max(np.abs(pred-again))<1e-14
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all() and pred.std()>0;np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),base_predictions=O,original_teacher=teachers['test'],stacked=pred)
  stem='submission_nested_stack_v5';csv=ROOT/(stem+'.csv');assert not csv.exists();sample.satisfaction=pred;sample.to_csv(csv,index=False);back=pd.read_csv(csv);assert list(back)==['id','satisfaction'] and len(back)==299844 and back.id.equals(test.id) and back.id.is_unique and back.satisfaction.notna().all() and back.satisfaction.between(0,1).all();np.testing.assert_allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
  rec.update(status='complete',meta_params=meta_params,meta_sha256=sha(out/'meta_residual_tree.txt'),meta_reload_maxdiff=float(np.max(np.abs(pred-again))),elapsed_seconds=time.monotonic()-start,ended_utc=now().isoformat(),submission={'path':str(csv),'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(back),'columns':list(back),'id_order_matches_official_test':True,'min':float(pred.min()),'max':float(pred.max()),'mean':float(pred.mean()),'std':float(pred.std()),'csv_readback_verified':True});dump(out/'result.json',rec);dump(ROOT/(stem+'_audit.json'),rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now().isoformat());dump(out/'result.json',rec);raise
if __name__=='__main__':main()

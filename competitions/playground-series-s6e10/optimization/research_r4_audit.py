"""Independent delivery audit of data/splits, own checkpoint lineage, weights and CSV."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 out=ROOT/'research_artifacts/r4_lean_cv_final';final=load(out/'result.json');assert final['status']=='complete';datahash={'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
 for name,h in datahash.items():assert sha(ROOT/'data'/f'{name}.csv')==h
 df=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);assert df.satisfaction.dtype==bool and raw.shape==(699635,21) and test.drop(columns='id').shape==(299844,21) and list(raw)==list(test.drop(columns='id'));assert df.id.is_unique and test.id.is_unique and not np.intersect1d(df.id,test.id).size
 records=[];derived={};base=ROOT/'research_artifacts/r4_baseline_replication3';b=np.load(base/'oof.npz');fold=b['fold'];expected=np.full(len(y),-1,dtype='int8')
 for k,(_,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261022).split(raw,y)):expected[vi]=k
 assert np.array_equal(fold,expected) and np.array_equal(b['id'],df.id) and np.array_equal(b['target'],y);P=np.zeros((len(y),5))
 for k in range(3):
  fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);tr=joblib.load(base/f'fold{k}/transformer.joblib');assert tr.n==len(fi);assert tr.counts.sort_index().equals(raw.iloc[fi].groupby('Flight Distance').size().sort_index())
  for j,name in enumerate(['lgb','cb8','residual','cb10','nn']):
   run=base/f'fold{k}'/name;rec=load(run/'result.json');assert rec['status']=='complete' and rec['model_sha256']==sha(run/rec['model_file']) and rec['prediction_sha256']==sha(run/'validation.npz');assert rec['train_indices_sha256']==hashlib.sha256(fi.tobytes()).hexdigest();z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);P[vi,j]=z['pred'];records.append({'case':'baseline3','fold':k,'component':name,'model_sha256':rec['model_sha256']})
   cfg=rec['config']
   if name=='nn':assert cfg['device']=='cpu' and cfg['n_ens']==2 and cfg['n_epochs']==3 and cfg['n_threads']==6
   elif name.startswith('cb'):assert cfg['iterations']==(1308 if name=='cb8' else 813) and cfg['task_type']=='CPU' and cfg['thread_count']==6
   else:assert cfg['n_estimators']==(639 if name=='lgb' else 581) and cfg['n_jobs']==6
 assert np.max(np.abs(P-b['components']))<1e-12;ref3=P@np.array([45,9,18,24,32])/128;assert np.max(np.abs(ref3-b['pred']))<1e-12
 for dirname,K,ref,old in [('r4_lean_gpu_init_oof',5,None,None),('r4_lean_gpu_init_replication3',3,ref3,P[:,4])]:
  path=ROOT/'research_artifacts'/dirname;res=load(path/'result.json');plan=load(path/'plan.json');assert res['status']=='complete' and res['eligible_for_new_partition_replication'];assert res['gpu_process_peak_bytes']<=1073741824 and res['oof_sha256']==sha(path/'oof.npz');z=np.load(path/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);f=z['fold'];nn=np.zeros(len(y))
  if K==5:
   ff=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(f,ff['fold']);legacy=np.load(ROOT/'research_artifacts/gpu_value_raw_oof_n2/oof.npz');ref=legacy['candidate'];old=legacy['pred']
  else:assert np.array_equal(f,fold)
  for k in range(K):
   fi=np.flatnonzero(f!=k);vi=np.flatnonzero(f==k);tr=joblib.load(path/f'fold{k}/transformer.joblib');assert abs(tr.encoder.target_mean_-float(y[fi].mean()))<1e-15
   for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
   members=[]
   for j,seed in enumerate(plan['seeds']):
    run=path/f'fold{k}/member{j}';r=load(run/'result.json');assert r['status']=='complete' and r['model_sha256']==sha(run/'model.pkl') and r['prediction_sha256']==sha(run/'validation.npz');assert r['config']==dict(plan['config'],random_state=seed);assert r['training_parameter_devices']==['cuda:0'] and len(r['epochs'])==3 and r['gpu_process_peak_bytes']<=1073741824;assert max(e['peak_allocated_bytes'] for e in r['epochs'])<=778043392;v=np.load(run/'validation.npz');assert np.array_equal(v['id'],df.id.iloc[vi]) and np.array_equal(v['target'],y[vi]);members.append(v['pred']);records.append({'case':dirname,'fold':k,'member':j,'model_sha256':r['model_sha256']})
   nn[vi]=np.mean(members,axis=0)
  assert np.max(np.abs(nn-z['pred']))<1e-12;candidate=ref+.25*(nn-old);err=float(np.max(np.abs(candidate-z['candidate'])));assert err<1e-12;derived[dirname]={'candidate_reconstruction_max_error':err,'auc':res['pooled']['candidate_auc'],'delta_auc':res['pooled']['delta_auc'],'conditional_95pct_ci':res['pooled']['conditional_95pct_ci'],'corrected_fold_lower':res['corrected_resampled_t']['lower_bound']}
 total=np.zeros(len(test));source=ROOT/'research_artifacts/r4_lean_gpu_init_oof'
 for k in range(5):
  for j in range(8):
   z=np.load(out/f'fold{k}_member{j}.npz');assert np.array_equal(z['id'],test.id) and np.isfinite(z['pred']).all();total+=z['pred']
 nn=total/40;components=np.load(out/'test_components.npz');assert np.array_equal(components['id'],test.id) and np.max(np.abs(nn-components['realmlp']))<1e-12;oldfull=np.load(ROOT/'research_artifacts/gpu_value_v6_full/test_components.npz');assert np.array_equal(oldfull['id'],test.id) and np.max(np.abs(oldfull['v5']-components['v5']))<1e-12;assert sha(ROOT/'submission_realmlp_value_v6.csv')=='36ea4ef80e1dd1911257f240391d9cdb8474b8b5deb5c52733582da80bdd1aaa';oldcsv=pd.read_csv(ROOT/'submission_realmlp_value_v6.csv');assert np.array_equal(oldcsv.id,test.id) and np.max(np.abs(oldcsv.satisfaction-oldfull['candidate']))<1e-14;core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz');assert np.array_equal(core['id'],test.id);corecheck=.46875*core['lgb_original']+.09375*core['cb_original']+.1875*core['residual_prior05']+.25*core['cb_deep'];assert np.max(np.abs(corecheck-components['v5']))<1e-12;candidate=.75*components['v5']+.25*nn;assert np.max(np.abs(candidate-components['candidate']))<1e-12;csv=ROOT/final['csv'];assert sha(csv)==final['sha256'];c=pd.read_csv(csv);assert list(c)==list(sample) and len(c)==299844 and np.array_equal(c.id,test.id) and np.array_equal(c.id,sample.id);err=float(np.max(np.abs(c.satisfaction-candidate)));assert err<1e-14 and np.isfinite(c.satisfaction).all() and c.satisfaction.between(0,1).all();assert final['gpu_process_peak_bytes']<=1073741824
 sourcecheckout=Path('/home/yang-zhi/Documents/Codex/2026-10-18/task/kaggle');original_clean=not subprocess.run(['git','-C',str(sourcecheckout),'status','--porcelain'],capture_output=True,text=True,check=True).stdout
 result={'status':'passed','audited_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':datahash,'train_rows':len(df),'test_rows':len(test),'raw_features':21,'positive_class':True,'new3fold_recomputed_seed':20261022,'all_baseline_components_freshly_refitted':True,'original_checkout_clean_at_audit':original_clean,'verified_model_records':len(records),'model_records':records,'validation':derived,'final_members':40,'weights':{'frozen_v5_full':.75,'new_lean_fivefold_mean':.25},'expanded_weights':{'LGB_original':45/128,'CB8':9/128,'residual_prior05':18/128,'CB10':24/128,'lean_cv40':32/128},'csv':csv.name,'sha256':sha(csv),'bytes':csv.stat().st_size,'csv_roundtrip_max_error':err,'no_unknown_test_predictions':True,'no_test_labels':True,'preprocessing_fit_on_training_rows_only':True,'source_implementation':'Publicrecipeparameter/mechanismreference only; our own models and predictions, officialpackages; no externalnotebookexecution','source_original_teacher':'Existingv5core uses previouslyaudited safeCC0 originaldata teacher/lookups, excludes21exactofficialoverlaps. No new externaldata inR4.','new_formal_submissions':0,'formal_total_limit':2,'actual_top100_verified':False,'selection_bias':'Officiallabels reused across historicalmodelsearch. These are fittingreplications andconditionalintervals, not untouchedholdout evidence.'};(ROOT/'research_r4_delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='model_records'},indent=2))
if __name__=='__main__':main()

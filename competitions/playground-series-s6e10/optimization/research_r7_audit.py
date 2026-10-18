"""Postqualification audit: exact fold membership, two cross-fit encoders, dual baselines and CSV."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,inspect
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from research_r4_common import evidence
from research_r7_response_features import ResponseFeatures
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 df=pd.read_csv(R/'data/train.csv');test=pd.read_csv(R/'data/test.csv');sample=pd.read_csv(R/'data/sample_submission.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);assert df.satisfaction.dtype==bool;hashes={'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
 for n,h in hashes.items():assert sha(R/'data'/f'{n}.csv')==h
 protocol=json.loads((R/'research_r7_plan.json').read_text())
 for n,h in protocol['code_sha256'].items():assert sha(R/n)==h
 validation={};models=[]
 for stage,K,seed,base in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
  out=R/'research_artifacts'/('r7_response_'+stage);rec=json.loads((out/'result.json').read_text());assert rec['eligible_for_new_partition_replication'] and rec['vs_private_half']['eligible_for_new_partition_replication'] and rec['vs_official_v7']['eligible_for_new_partition_replication'];z=np.load(out/'oof.npz');old=np.load(R/'research_artifacts'/base/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],old['fold']);fold=z['fold'];expected=np.full(len(y),-1,'int8')
  for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):expected[vi]=k
  assert np.array_equal(fold,expected);pred=np.zeros(len(y));plan=json.loads((out/'plan.json').read_text())
  for k in range(K):
   fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);tr=joblib.load(out/f'fold{k}/transformer.joblib');assert abs(tr.encoder.target_mean_-y[fi].mean())<1e-15 and abs(tr.response_encoder.target_mean_-y[fi].mean())<1e-15;assert tr.encoder.cv==5 and tr.response_encoder.cv==5 and tr.response_encoder.random_state==20261025
   for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
   keys=tr.response_keys(raw.iloc[fi].reset_index(drop=True));assert len(tr.response_encoder.categories_)==13
   for j,c in enumerate(keys):assert set(tr.response_encoder.categories_[j])==set(keys[c])
   m=[]
   for j,seedj in enumerate(plan['seeds']):
    path=out/f'fold{k}/member{j}';r=json.loads((path/'result.json').read_text());assert r['status']=='complete' and r['model_sha256']==sha(path/'model.pkl') and r['prediction_sha256']==sha(path/'validation.npz');assert r['config']==dict(plan['config'],random_state=seedj) and len(r['epochs'])==3 and r['training_parameter_devices']==['cuda:0'] and r['gpu_process_peak_bytes']<=1073741824;assert max(e['peak_allocated_bytes'] for e in r['epochs'])<=778043392;v=np.load(path/'validation.npz');assert np.array_equal(v['id'],df.id.iloc[vi]) and np.array_equal(v['target'],y[vi]);m.append(v['pred']);models.append({'stage':stage,'fold':k,'member':j,'sha256':r['model_sha256']})
   pred[vi]=np.mean(m,axis=0)
  assert np.max(np.abs(pred-z['pred']))<1e-12;v5=(old['candidate']-.25*old['pred'])/.75;half=.5*v5+.5*old['pred'];candidate=.5*v5+.5*pred;assert np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(half-z['reference_half']))<1e-12
  direct=evidence(y,old['candidate'],candidate,fold);control=evidence(y,half,candidate,fold);assert direct['eligible_for_new_partition_replication'] and control['eligible_for_new_partition_replication'];validation[stage]={'vs_v7':direct,'vs_half_control':control}
 out=R/'research_artifacts/r7_response_cv_final';final=json.loads((out/'result.json').read_text());z=np.load(out/'test_components.npz');assert np.array_equal(z['id'],test.id);acc=np.zeros(len(test))
 for k in range(5):
  for j in range(8):
   a=np.load(out/f'fold{k}_member{j}.npz');assert np.array_equal(a['id'],test.id);acc+=a['pred']
 assert np.max(np.abs(acc/40-z['realmlp']))<1e-12;old=np.load(R/'research_artifacts/r4_lean_cv_final/test_components.npz');assert np.array_equal(old['id'],test.id) and np.max(np.abs(old['v5']-z['v5']))<1e-12;pred=.5*z['v5']+.5*z['realmlp'];csv=R/final['csv'];assert sha(csv)==final['sha256'];c=pd.read_csv(csv);assert list(c)==list(sample) and len(c)==299844 and np.array_equal(c.id,test.id) and np.array_equal(c.id,sample.id);err=float(np.max(np.abs(c.satisfaction-pred)));assert err<1e-14 and c.satisfaction.between(0,1).all()
 result={'status':'passed','audited_utc':datetime.now(timezone.utc).isoformat(),'csv':csv.name,'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(c),'positive_class':True,'data_sha256':hashes,'validation':validation,'verified_model_records':len(models),'model_records':models,'fixed_weights':{'v5_full':.5,'responseNN_cv40':.5},'raw_rating_categories_kept':True,'response_numeric_channels':13,'input_width':47,'numeric_width':25,'sklearn_fit_transform_source_sha256':hashlib.sha256(inspect.getsource(TargetEncoder.fit_transform).encode()).hexdigest(),'outer_training_counts_and_target_means_verified':True,'inner_folds_target_means':'Installedofficialsklearnfit_transformcomputesy_trainmeanineachCVfold; outervalidationlabelsneverpassedtogenerator.','CSV_roundtrip_error':err,'formal_used':1,'remaining':1,'actualTop100verified':False,'source':'OfficialdataonlynewNN; existingauditedCC0core, ownmodels/predictions, no newexternaldata/deps.','selection_bias':'Oldlabelsreused andprivatecontrolselectedhistorically; no pristineholdoutclaim.'};(R/'research_r7_delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['model_records','validation']},indent=2))
if __name__=='__main__':main()

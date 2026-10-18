"""Full refit/export only after the capped GPU numeric-value view passes all frozen gates."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,signal
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits,threadpool_info
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_gpu_value_features import ValueFeatures
from research_gpu_value import setup_gpu,save_model,load_model
from research_external_ablation import external
from research_original import safe_original
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
WEIGHTS={'v5':.75,'realmlp':.25}
def now():return datetime.now(timezone.utc)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((ROOT/'research_gpu_value_plan.json').read_text());selection=json.loads((ROOT/'research_gpu_value_selection.json').read_text());assert selection['eligible_for_full'];variant=selection['chosen_variant'];n=selection['n_ens'];primary=selection['primary'][variant];cfg=dict(plan['config'],n_ens=n);assert selection['weights']==WEIGHTS and selection['epochs']==3;assert primary['vs_v5']['delta_auc']>=.0001 and primary['vs_v5']['conditional_95pct_ci'][0]>0 and selection['confirmation']['vs_v5']['conditional_95pct_ci'][0]>0
 ledger=json.loads((ROOT/'research_submission_ledger.json').read_text());assert ledger['completed_new_submissions']==3 and ledger['remaining_unreserved_slots']==1
 limits=plan['limits'];assert now()<datetime.fromisoformat(limits['training_start_cutoff_utc']);deadline=datetime.fromisoformat(limits['training_hard_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-now()).total_seconds())))
 initial_gpu_free=setup_gpu(plan)
 out=ROOT/'research_artifacts/gpu_value_v6_full';out.mkdir(exist_ok=False);start=time.monotonic();rec={'status':'preflight','started_utc':now().isoformat(),'weights':WEIGHTS,'plan_sha256':sha(ROOT/'research_gpu_value_plan.json'),'runner_sha256':sha(__file__),'epochs':[],'variant':variant,'n_ens':n,'initial_gpu_free_bytes':initial_gpu_free};dump(out/'result.json',rec)
 try:
  expected={'train.csv':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test.csv':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission.csv':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'};assert all(sha(ROOT/'data'/n)==v for n,v in expected.items());rec['data_sha256']=expected
  original,train,test,cols,audit=safe_original();del original;rec['original_source_audit']=audit;y=train.satisfaction.astype('int8').to_numpy();sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert list(sample)==['id','satisfaction'] and sample.id.equals(test.id) and len(test)==299844
  oofpath=ROOT/'research_artifacts'/f'gpu_value_{variant}_oof_n{n}';oofplan=json.loads((oofpath/'plan.json').read_text());assert oofplan['frozen_protocol']==plan and oofplan['runner_sha256']==sha(ROOT/'research_gpu_value.py') and oofplan['feature_sha256']==sha(ROOT/'research_gpu_value_features.py');assert oofplan['external_features_sha256']==sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')
  o=np.load(oofpath/'oof.npz');v=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert all(np.array_equal(o[k],v[k]) for k in ['id','target','fold']);assert np.array_equal(o['id'],train.id) and np.array_equal(o['target'],y);candidate=.75*v['pred']+.25*o['pred'];assert np.array_equal(candidate,o['candidate']);comp=compare(y,v['pred'],candidate);foldcomp=[compare(y[o['fold']==k],v['pred'][o['fold']==k],candidate[o['fold']==k]) for k in range(5)];assert comp['delta_auc']>=.0002 and comp['conditional_95pct_ci'][0]>0 and all(c['delta_auc']>0 for c in foldcomp);rec.update(oof_comparison=comp,oof_fold_comparisons=foldcomp,oof_sha256=sha(oofpath/'oof.npz'))
  oldcsv=ROOT/'submission_cb_deep_v5.csv';assert sha(oldcsv)=='0f9a978051e4f0fa164928f3b975a71346b60477959ee5cf50abb580a706e872';old=pd.read_csv(oldcsv);parts=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz');assert old.id.equals(test.id) and np.array_equal(parts['id'],test.id);base=parts['candidate'];assert np.allclose(base,old.satisfaction,rtol=0,atol=1e-14)
  ext,_=external('both',train.id);te=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz');assert np.array_equal(te['id'],test.id) and list(te['columns'])==list(ext.columns);tex=pd.DataFrame(te['values'],columns=te['columns']);tr=ValueFeatures(variant);X=tr.fit_transform(train[cols],y,ext);T=tr.transform(test[cols],tex);joblib.dump(tr,out/'transformer.joblib');rec.update(status='training',feature_width=X.shape[1],train_rows=len(train),test_rows=len(test),config=cfg);dump(out/'result.json',rec)
  model=RealMLP_TD_Classifier(**cfg);assert model.get_config()==oofplan['resolved_config'];original_hook=TabNNModule.on_train_epoch_end;last=[time.monotonic()]
  def hook(module):
   original_hook(module);torch.cuda.synchronize();t=time.monotonic();free,_=torch.cuda.mem_get_info();assert free>=1024**3,'GPU free-memory headroom guard';row={'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'utc':now().isoformat(),'gpu_free_bytes':free};last[0]=t;rec['epochs'].append(row);dump(out/'result.json',rec);print('full',row,flush=True)
  TabNNModule.on_train_epoch_end=hook
  with threadpool_limits(6):
   rec['threadpools']=threadpool_info();fitstart=time.monotonic()
   try:model.fit(X,y)
   finally:TabNNModule.on_train_epoch_end=original_hook
   rec['fit_seconds']=time.monotonic()-fitstart;assert len(rec['epochs'])==3 and model.cv_alg_interface_.device=='cuda:0' and not model.cv_alg_interface_.model.creator.is_cv and np.array_equal(model.classes_,[0,1]);pred=model.predict_proba(T)[:,1];model.to('cpu');save_model(model,out/'model.pkl');loaded=load_model(out/'model.pkl');loaded.to('cuda:0');again=loaded.predict_proba(T)[:,1];diff=float(np.max(np.abs(pred-again)));assert diff<1e-7
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=.75*base+.25*pred;np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v5=base,realmlp=pred,candidate=candidate);sample.satisfaction=candidate;csv=ROOT/'submission_realmlp_value_v6.csv';assert not csv.exists();sample.to_csv(csv,index=False);back=pd.read_csv(csv);assert list(back)==['id','satisfaction'] and len(back)==299844 and back.id.equals(test.id) and back.id.is_unique and np.isfinite(back.satisfaction).all() and back.satisfaction.between(0,1).all();csvdiff=float(np.max(np.abs(back.satisfaction-candidate)));assert csvdiff<1e-14
  rec.update(status='complete',actual_epochs=3,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_sha256=sha(out/'model.pkl'),transformer_sha256=sha(out/'transformer.joblib'),model_reload_max_abs_diff=diff,csv_component_max_abs_diff=csvdiff,completed_utc=now().isoformat(),elapsed_seconds=time.monotonic()-start,submission={'path':str(csv),'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(back),'columns':list(back),'id_order_matches_official':True});dump(out/'result.json',rec);dump(ROOT/'submission_realmlp_value_v6_audit.json',rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now().isoformat());dump(out/'result.json',rec);raise
if __name__=='__main__':main()

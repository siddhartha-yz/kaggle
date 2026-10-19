"""One frozen official PyTabKit RealMLP-TD recipe; train-only preprocessing and CPU6."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,argparse,signal,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,torch,joblib
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits,threadpool_info
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_realmlp_features import RealMLPFeatures
from research_external_ablation import external
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=p.parse_args();plan=json.loads((ROOT/'research_realmlp_plan.json').read_text());limits=plan['limits'];deadline=datetime.fromisoformat(limits['training_hard_cutoff_utc']);torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.version.cuda is None
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))
 out=ROOT/'research_artifacts'/f'realmlp48_{a.split}';out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id)
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];rz=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert np.array_equal(rz['id'],df.id) and np.array_equal(rz['target'],y) and np.array_equal(rz['fold'],fold);ref=rz['pred']
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])];vi=z['valid'];rz=np.load(ROOT/'research_artifacts'/f'cb_deep_{a.split}'/'candidate_w0.25.npz');assert np.array_equal(rz['idx'],vi) and np.array_equal(rz['id'],df.id.iloc[vi]) and np.array_equal(rz['target'],y[vi]);ref=rz['pred']
 runplan={'protocol':plan,'runner_sha256':sha(__file__),'feature_sha256':sha(ROOT/'research_realmlp_features.py'),'reference_sha256':sha(rz.zip.filename),'external_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')}
 if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==runplan
 else:dump(out/'plan.json',runplan)
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(limits['training_start_cutoff_utc'])
  start=time.monotonic();rec={'status':'preprocessing','started_utc':now(),'epochs':[],'train_rows':len(fi),'valid_rows':len(vi)};dump(run/'result.json',rec)
  try:
   tr=RealMLPFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib');rec.update(status='training',feature_width=X.shape[1],preprocessing_seconds=time.monotonic()-start);dump(run/'result.json',rec)
   model=RealMLP_TD_Classifier(**plan['config']);assert model.get_config()==plan['resolved_config'];last=[time.monotonic()];original_hook=TabNNModule.on_train_epoch_end
   def epoch_hook(module):
    original_hook(module);t=time.monotonic();row={'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'utc':now()};last[0]=t;rec['epochs'].append(row);dump(run/'result.json',rec);print(a.split,k,row,flush=True)
   TabNNModule.on_train_epoch_end=epoch_hook
   with threadpool_limits(6):
    rec['threadpools']=threadpool_info();fitstart=time.monotonic()
    try:model.fit(X,y[fi])
    finally:TabNNModule.on_train_epoch_end=original_hook
    fitseconds=time.monotonic()-fitstart;assert len(rec['epochs'])==plan['config']['n_epochs'];assert not model.cv_alg_interface_.model.creator.is_cv;assert model.cv_alg_interface_.device=='cpu';assert np.array_equal(model.classes_,[0,1])
    pred=model.predict_proba(V)[:,1];joblib.dump(model,run/'model.joblib');reloaded=joblib.load(run/'model.joblib');check=reloaded.predict_proba(V.iloc[:1024])[:,1];reload_diff=float(np.max(np.abs(check-pred[:1024])));assert reload_diff<1e-7
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();localref=ref[vi] if a.split=='oof' else ref;candidate=.75*localref+.25*pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate)
   rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),vs_v5=compare(y[vi],localref,candidate),fit_seconds=fitseconds,reload_max_abs_diff=reload_diff,model_sha256=sha(run/'model.joblib'),elapsed_seconds=time.monotonic()-start,ended_utc=now(),actual_epochs=len(rec['epochs']),validation_passed_to_fit=False);dump(run/'result.json',rec);del model,reloaded,X,V;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):
   z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  target=y;ids=df.id.to_numpy();idx=np.arange(len(y));np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred,candidate=.75*ref+.25*pred)
 else:
  z=np.load(out/'fold0/validation.npz');pred=z['pred'];idx=z['idx'];target=y[idx];ids=df.id.iloc[idx].to_numpy()
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred,candidate=.75*ref+.25*pred)
 result={'status':'complete','split':a.split,'auc':float(roc_auc_score(target,pred)),'vs_v5':compare(target,ref,.75*ref+.25*pred),'fold_comparisons':[json.loads((out/f'fold{k}/result.json').read_text())['vs_v5'] for k in range(len(splits))],'completed_utc':now()};dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

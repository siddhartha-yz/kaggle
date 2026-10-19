"""Official RealMLP value-embedding views; capped GPU and fixed validation protocol."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,argparse,signal,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,torch,joblib,cloudpickle
from threadpoolctl import threadpool_limits,threadpool_info
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_gpu_value_features import ValueFeatures
from research_external_ablation import external
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup_gpu(plan):
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.version.cuda=='12.8';props=torch.cuda.get_device_properties(0);cap=plan['gpu_allocator_cap_bytes'];free,_=torch.cuda.mem_get_info();assert free>=cap+int(1.25*1024**3),'GPU initial headroom guard';torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();return free

def save_model(model,path):
 with Path(path).open('wb') as f:cloudpickle.dump(model,f)
def load_model(path):
 with Path(path).open('rb') as f:return cloudpickle.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);p.add_argument('--variant',choices=['raw','context'],required=True);p.add_argument('--n-ens',type=int,choices=[1,2],required=True);a=p.parse_args();plan=json.loads((ROOT/'research_gpu_value_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc'])
 def alarm(s,f):raise TimeoutError('Frozen training cutoff')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))
 name=f'gpu_value_{a.variant}_{a.split}_n{a.n_ens}';out=ROOT/'research_artifacts'/name;out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id);fold=None
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');fold=z['fold'];assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];r=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert all(np.array_equal(r[k],z[k]) for k in ['id','target','fold']);ref=r['pred']
 else:
  z=np.load(ROOT/('iteration_artifacts/split_confirm20261017.npz' if a.split=='confirm' else 'artifacts/split_seed42.npz'));splits=[(z['train'],z['valid'])];vi=z['valid'];r=np.load(ROOT/'research_artifacts'/f"cb_deep_{'confirm' if a.split=='confirm' else 'primary'}"/'candidate_w0.25.npz');assert np.array_equal(r['idx'],vi) and np.array_equal(r['id'],df.id.iloc[vi]) and np.array_equal(r['target'],y[vi]);ref=r['pred']
 cfg=dict(plan['config']);cfg['n_ens']=a.n_ens
 if a.split=='probe':cfg['n_epochs']=1
 runplan={'frozen_protocol':plan,'config':cfg,'resolved_config':RealMLP_TD_Classifier(**cfg).get_config(),'runner_sha256':sha(__file__),'feature_sha256':sha(ROOT/'research_gpu_value_features.py'),'reference_sha256':sha(r.zip.filename),'source_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')}
 if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==runplan
 else:dump(out/'plan.json',runplan)
 initial_free=setup_gpu(plan)
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(plan['limits']['training_start_cutoff_utc']);start=time.monotonic();rec={'status':'preprocessing','started_utc':now(),'epochs':[],'train_rows':len(fi),'valid_rows':len(vi),'initial_gpu_free_bytes':initial_free};dump(run/'result.json',rec)
  try:
   free_now,_=torch.cuda.mem_get_info();assert free_now>=plan['gpu_allocator_cap_bytes']+int(1.25*1024**3),'GPU fold-start headroom guard'
   tr=ValueFeatures(a.variant);X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=None if a.split=='probe' else tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib');m=RealMLP_TD_Classifier(**cfg);hook=TabNNModule.on_train_epoch_end;last=[time.monotonic()];rec.update(status='training',feature_width=X.shape[1]);dump(run/'result.json',rec)
   def epoch_hook(module):
    hook(module);torch.cuda.synchronize();t=time.monotonic();free,_=torch.cuda.mem_get_info();row={'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'gpu_free_bytes':free};last[0]=t;rec['epochs'].append(row);dump(run/'result.json',rec);print(name,k,row,flush=True)
    if free<1024**3:raise RuntimeError('GPU free-memory headroom below1GiB')
   TabNNModule.on_train_epoch_end=epoch_hook
   with threadpool_limits(6):
    rec['threadpools']=threadpool_info();fitstart=time.monotonic()
    try:m.fit(X,y[fi])
    finally:TabNNModule.on_train_epoch_end=hook
    rec.update(fit_seconds=time.monotonic()-fitstart,actual_epochs=len(rec['epochs']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());assert len(rec['epochs'])==cfg['n_epochs'] and not m.cv_alg_interface_.model.creator.is_cv and np.array_equal(m.classes_,[0,1])
    if a.split=='probe':
     probe=m.predict_proba(X.iloc[:1024])[:,1];assert np.isfinite(probe).all();rec.update(status='complete',score_evaluated=False,ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);return
    pred=m.predict_proba(V)[:,1];m.to('cpu');save_model(m,run/'model.pkl');loaded=load_model(run/'model.pkl');loaded.to('cuda:0');again=loaded.predict_proba(V)[:,1];diff=float(np.max(np.abs(again-pred)));assert diff<1e-7
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();localref=ref[vi] if a.split=='oof' else ref;candidate=.75*localref+.25*pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate);rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),vs_v5=compare(y[vi],localref,candidate),model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.pkl'),elapsed_seconds=time.monotonic()-start,ended_utc=now());dump(run/'result.json',rec);del m,loaded,X,V;gc.collect();torch.cuda.empty_cache()
   if a.split=='oof' and rec['vs_v5']['delta_auc']<=0:
    dump(out/'result.json',{'status':'complete','split':'oof','rejected_reason':'Frozen every-fold-positive gate failed','completed_folds':k+1,'eligible_for_full':False,'fold_comparisons':[json.loads((out/f'fold{j}/result.json').read_text())['vs_v5'] for j in range(k+1)]});return
  except BaseException as e:rec.update(status='failed',error=repr(e),hardware_oom=isinstance(e,torch.OutOfMemoryError),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  target=y;idx=np.arange(len(y));ids=df.id.to_numpy();np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred,candidate=.75*ref+.25*pred)
 else:
  z=np.load(out/'fold0/validation.npz');pred=z['pred'];idx=z['idx'];target=y[idx];ids=z['id']
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred,candidate=.75*ref+.25*pred);result={'status':'complete','name':name,'split':a.split,'variant':a.variant,'n_ens':a.n_ens,'auc':float(roc_auc_score(target,pred)),'vs_v5':compare(target,ref,.75*ref+.25*pred),'fold_comparisons':[json.loads((out/f'fold{k}/result.json').read_text())['vs_v5'] for k in range(len(splits))],'completed_utc':now()};dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

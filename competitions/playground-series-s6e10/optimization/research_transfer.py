"""Frozen source-pretraining ablation, selected for a fixed blend against v5."""
from pathlib import Path
from datetime import datetime, timezone
import os,json,time,hashlib,argparse,signal,gc
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='6'
import numpy as np,pandas as pd,joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits,threadpool_info
from research_neural_features import NeuralFeatures
from research_original import safe_original
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def empty(n):return pd.DataFrame(index=np.arange(n))
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','confirm','oof'],required=True);p.add_argument('--pretrain',type=int,choices=[0,8],required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--fixed',action='store_true');a=p.parse_args();assert a.split=='primary' or a.fixed
 plan=json.loads((ROOT/'research_transfer_plan.json').read_text());limits=plan['limits'];deadline=datetime.fromisoformat(limits['training_hard_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))
 name=f'nn_transfer_p{a.pretrain}_{a.split}'+(f'_e{a.epochs}' if a.fixed else '');out=ROOT/'research_artifacts'/name;out.mkdir(exist_ok=True)
 original,df,test,cols,audit=safe_original();del test
 y=df.satisfaction.astype('int8').to_numpy();sy=original.satisfaction.astype('int8').to_numpy();raw=df[cols];source=original[cols]
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];rz=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert np.array_equal(rz['id'],df.id) and np.array_equal(rz['target'],y) and np.array_equal(rz['fold'],fold);ref=rz['pred']
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])];vi=z['valid'];rz=np.load(ROOT/'research_artifacts'/f'cb_deep_{a.split}'/'candidate_w0.25.npz');assert np.array_equal(rz['idx'],vi) and np.array_equal(rz['id'],df.id.iloc[vi]) and np.array_equal(rz['target'],y[vi]);ref=rz['pred']
 runplan={'protocol':plan,'source_audit':audit,'reference_sha256':sha(rz.zip.filename),'runner_sha256':sha(__file__),'feature_sha256':sha(ROOT/'research_neural_features.py'),'pretrain_epochs':a.pretrain,'fine_tune_epochs':a.epochs,'fixed':a.fixed,'started_utc':now()}
 if (out/'plan.json').exists():
  old=json.loads((out/'plan.json').read_text());assert all(old[k]==runplan[k] for k in ['runner_sha256','feature_sha256','reference_sha256','pretrain_epochs','fine_tune_epochs','fixed'])
 else:dump(out/'plan.json',runplan)
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(limits['training_start_cutoff_utc'])
  start=time.monotonic();rec={'status':'preprocessing','started_utc':now(),'pretrain_epochs':[],'epochs':[]};dump(run/'result.json',rec)
  try:
   tr=NeuralFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],empty(len(fi)));V=tr.transform(raw.iloc[vi].reset_index(drop=True),empty(len(vi)));S=tr.transform(source,empty(len(source))) if a.pretrain else None;joblib.dump(tr,run/'transformer.joblib')
   localref=ref[vi] if a.split=='oof' else ref
   config=dict(plan['network']);config['hidden_layer_sizes']=tuple(config['hidden_layer_sizes']);model=MLPClassifier(**config,random_state=np.random.RandomState(plan['seed']));best=-1.;bestepoch=0;wait=0
   rec.update(status='training',feature_width=X.shape[1],preprocessing_seconds=time.monotonic()-start);dump(run/'result.json',rec)
   with threadpool_limits(limits=6):
    rec['threadpools']=threadpool_info()
    for epoch in range(1,a.pretrain+1):
     e=time.monotonic();model.partial_fit(S,sy,classes=[0,1]);row={'epoch':epoch,'training_seconds':time.monotonic()-e,'training_loss':float(model.loss_)};rec['pretrain_epochs'].append(row);dump(run/'result.json',rec);print(name,k,'source',row,flush=True)
    for epoch in range(1,a.epochs+1):
     e=time.monotonic();model.partial_fit(X,y[fi],classes=[0,1]);seconds=time.monotonic()-e;row={'epoch':epoch,'training_seconds':seconds,'training_loss':float(model.loss_)}
     if not a.fixed:
      pred=model.predict_proba(V)[:,1];auc=float(roc_auc_score(y[vi],.75*localref+.25*pred));row.update(validation_auc=float(roc_auc_score(y[vi],pred)),blend_auc=auc)
      if auc>best+1e-6:best=auc;bestepoch=epoch;wait=0;joblib.dump(model,run/'model.joblib')
      else:wait+=1
     rec['epochs'].append(row);dump(run/'result.json',rec);print(name,k,'synthetic',row,flush=True)
     if not a.fixed and ((epoch==1 and seconds>180) or wait>=plan['primary_patience'] or time.monotonic()-start>plan['primary_seconds_cap']):break
    if a.fixed:bestepoch=a.epochs;joblib.dump(model,run/'model.joblib')
    else:model=joblib.load(run/'model.joblib')
    pred=model.predict_proba(V)[:,1]
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=.75*localref+.25*pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate)
   rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),vs_v5=compare(y[vi],localref,candidate),selected_epochs=bestepoch,model_sha256=sha(run/'model.joblib'),elapsed_seconds=time.monotonic()-start,ended_utc=now());dump(run/'result.json',rec);del model,X,V,S;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):
   z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  target=y;ids=df.id.to_numpy();idx=np.arange(len(y));np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred,candidate=.75*ref+.25*pred)
 else:
  z=np.load(out/'fold0/validation.npz');pred=z['pred'];idx=z['idx'];target=y[idx];ids=df.id.iloc[idx].to_numpy()
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred,candidate=.75*ref+.25*pred)
 result={'status':'complete','name':name,'auc':float(roc_auc_score(target,pred)),'vs_v5':compare(target,ref,.75*ref+.25*pred),'selected_epochs':[json.loads((out/f'fold{k}/result.json').read_text())['selected_epochs'] for k in range(len(splits))],'fold_comparisons':[json.loads((out/f'fold{k}/result.json').read_text())['vs_v5'] for k in range(len(splits))],'completed_utc':now()};dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

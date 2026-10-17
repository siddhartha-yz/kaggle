"""Single prespecified sklearn MLP, CPU6, bounded primary epoch selection or fixed fits."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,argparse,signal,gc
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='6';os.environ['MKL_NUM_THREADS']='6';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits,threadpool_info
from research_neural_features import NeuralFeatures
from research_external_ablation import external
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','confirm','oof'],default='primary');p.add_argument('--epochs',type=int,default=40);p.add_argument('--fixed',action='store_true');p.add_argument('--variant',choices=['v1','v2'],default='v1');a=p.parse_args();assert a.split=='primary' or a.fixed
 protocol=json.loads((ROOT/'research_protocol.json').read_text());deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard training cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-datetime.now(timezone.utc)).total_seconds())))
 name=f'neural_{a.variant}_{a.split}'+(f'_e{a.epochs}' if a.fixed else '');out=ROOT/'research_artifacts'/name;out.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id)
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)];ref=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz')['pred']
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])];ref=np.load(ROOT/'research_artifacts'/f'e1_original_both_{a.split}'/'validation.npz')['pred']
 config=dict(hidden_layer_sizes=(256,128,64),activation='relu',solver='adam',alpha=.001,batch_size=2048,learning_rate_init=.001,max_iter=1,shuffle=True,early_stopping=False)
 patience=7
 if a.variant=='v2':config.update(hidden_layer_sizes=(512,256,128),alpha=.1,learning_rate_init=.0003);patience=10
 plan={'name':name,'config':config,'seed':42,'maximum_epochs':a.epochs,'fixed_epochs':a.fixed,'primary_patience':patience,'primary_time_cap_seconds':900,'first_epoch_abort_seconds':180,'cpu_threads':6,'preprocessing_sha256':sha(ROOT/'research_neural_features.py'),'runner_sha256':sha(__file__),'external_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'frozen_utc':now()}
 if not (out/'plan.json').exists():dump(out/'plan.json',plan)
 else:
  old=json.loads((out/'plan.json').read_text());assert old['preprocessing_sha256']==plan['preprocessing_sha256'] and old['maximum_epochs']==a.epochs and old['fixed_epochs']==a.fixed
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc'])
  start=time.monotonic();rec={'status':'preprocessing','fold':k,'started_utc':now(),'epochs':[]};dump(run/'result.json',rec)
  try:
   tr=NeuralFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
   model=MLPClassifier(**config,random_state=np.random.RandomState(42));best=-1.;bestepoch=0;wait=0;rec.update(status='training',feature_width=X.shape[1],preprocessing_seconds=time.monotonic()-start);dump(run/'result.json',rec)
   with threadpool_limits(limits=6):
    rec['threadpools']=threadpool_info()
    for epoch in range(1,a.epochs+1):
     e=time.monotonic();model.partial_fit(X,y[fi],classes=[0,1]);seconds=time.monotonic()-e;row={'epoch':epoch,'training_seconds':seconds,'training_loss':float(model.loss_)}
     if not a.fixed:
      pred=model.predict_proba(V)[:,1];auc=float(roc_auc_score(y[vi],pred));row['validation_auc']=auc
      if auc>best+1e-6:best=auc;bestepoch=epoch;wait=0;joblib.dump(model,run/'model.joblib')
      else:wait+=1
     rec['epochs'].append(row);dump(run/'result.json',rec);print(name,k,row,flush=True)
     if not a.fixed and ((epoch==1 and seconds>180) or wait>=patience or time.monotonic()-start>900):break
    if a.fixed:bestepoch=a.epochs;joblib.dump(model,run/'model.joblib')
    else:model=joblib.load(run/'model.joblib')
    pred=model.predict_proba(V)[:,1]
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred)
   rec.update(status='complete',auc=float(roc_auc_score(y[vi],pred)),logloss=float(log_loss(y[vi],pred)),selected_epochs=bestepoch,model_sha256=sha(run/'model.joblib'),elapsed_seconds=time.monotonic()-start,ended_utc=now());dump(run/'result.json',rec);del model,X,V;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):
   z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  target=y;ids=df.id.to_numpy();idx=np.arange(len(y));np.savez_compressed(out/'oof.npz',id=ids,target=y,fold=fold,pred=pred)
 else:
  z=np.load(out/'fold0/validation.npz');pred=z['pred'];idx=z['idx'];target=y[idx];ids=df.id.iloc[idx].to_numpy()
 np.savez_compressed(out/'validation.npz',idx=idx,id=ids,target=target,pred=pred)
 result={'status':'complete','name':name,'auc':float(roc_auc_score(target,pred)),'vs_original_lgb':compare(target,ref,pred),'fixed_grid_blends':{str(w):compare(target,ref,(1-w)*ref+w*pred) for w in [.25,.5]},'selected_epochs':[json.loads((out/f'fold{k}/result.json').read_text())['selected_epochs'] for k in range(len(splits))],'completed_utc':now()};dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

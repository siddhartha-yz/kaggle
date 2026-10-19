"""One fixed raw21 EBM outer fold; bag validation uses own-training only."""
from research_r77_model import ROOT,make_model,parameters,reconstruct_probability,trace_inner_bags
from research_r77_probe import watch
import os,json,sys,time,threading,hashlib,resource,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedKFold

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);temp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,p)
def state(status,**kw):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),outer_quality_scored=False,GPU_used=False,**kw);dump(ROOT/'research_r77_state.json',d);print(json.dumps(d),flush=True)
def save_npz(p,**arrays):
 if p.exists():
  with np.load(p) as old:assert set(old.files)==set(arrays) and all(np.array_equal(old[n],v) for n,v in arrays.items())
  return
 temp=p.with_name('.'+p.stem+f'.{os.getpid()}.npz');np.savez_compressed(temp,**arrays);os.replace(temp,p)

def main(stage,k):
 plan=json.loads((ROOT/'research_r77_plan.json').read_text());ph=sha(ROOT/'research_r77_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r77_ebm_raw21_fivefold/result.json').read_text())['eligible']
 run=ROOT/f'research_artifacts/r77_ebm_raw21_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'result.json').exists():
  old=json.loads((run/'result.json').read_text());assert old['status']=='complete' and old['plan_sha256']==ph and all(sha(run/n)==h for n,h in old['files_sha256'].items());return
 available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;assert available>=6*2**30
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True).start();tick=time.monotonic();active=threading.Event();active.set()
 def heartbeat():
  while not stop.wait(30):
   if active.is_set():state('own_training_EBM_fit',stage=stage,fold=k,seconds=time.monotonic()-tick,process_CPU_seconds=time.process_time(),peak_RSS_bytes=peak[0],outer_bags=8,max_rounds=50000)
 threading.Thread(target=heartbeat,daemon=True).start()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');assert len(frame)==699635 and frame.satisfaction.dtype==bool and frame.id.is_unique
  ids=frame.id.to_numpy(copy=True);y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);assert raw.shape==(699635,21);base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(ids,base['id']) and np.array_equal(y,base['target'])
  K,splitseed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(y),-1,dtype='int8')
  for j,(_,vi0) in enumerate(StratifiedKFold(K,shuffle=True,random_state=splitseed).split(raw,y)):fold[vi0]=j
  assert np.array_equal(fold,base['fold']);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);X=raw.iloc[fi].reset_index(drop=True);V=raw.iloc[vi].reset_index(drop=True);assert len(X.select_dtypes(exclude='number').columns)==4
  save_npz(run/'outer_indices.npz',fi=fi,vi=vi,train_id=ids[fi],valid_id=ids[vi]);del frame,raw;gc.collect()
  ckpt=run/'model.joblib';metadata=run/'training.json'
  if metadata.exists():
   fit=json.loads(metadata.read_text());assert fit['plan_sha256']==ph and sha(ckpt)==fit['model_sha256'];m=joblib.load(ckpt);state('restored_completed_EBM_fit',stage=stage,fold=k)
  else:
   m=make_model(X);assert all(m.get_params()[n]==v for n,v in plan['parameters'].items());state('own_training_EBM_fit',stage=stage,fold=k,training_rows=len(fi),validation_rows=len(vi),outer_bags=8,max_rounds=50000)
   with trace_inner_bags(y[fi]) as bags:
    with joblib.parallel_backend('threading',n_jobs=8):
     assert type(joblib.Parallel(n_jobs=8)._backend).__name__=='ThreadingBackend';m.fit(X,y[fi])
   assert len(bags)==8;save_npz(run/'inner_bags.npz',bag_matrix=np.stack([b['bag'] for b in bags]),rng_before=np.stack([b['rng_before'] for b in bags]),rng_after=np.stack([b['rng_after'] for b in bags]));del bags
   m.callback=None;temp=run/'.model.joblib.tmp';joblib.dump(m,temp);os.replace(temp,ckpt)
   fit=dict(plan_sha256=ph,model_sha256=sha(ckpt),inner_bags_sha256=sha(run/'inner_bags.npz'),training_rows=len(fi),raw_features=list(X),nominal_features=list(X.select_dtypes(exclude='number')),parameters=plan['parameters'],best_iteration=np.asarray(m.best_iteration_).tolist(),terms=[list(a) for a in m.term_features_],outer_validation_used=False,backend='ThreadingBackend',inner_validation='Eight native85/15 own-outer-training bag splits, seed42. No outervalidation data passed to fit. Native earlystop100/tolerance1e-5. Original native RNG and memberships observed without consuming randomness or changing return values.',fit_seconds=time.monotonic()-tick);dump(metadata,fit)
  active.clear();assert np.array_equal(m.classes_,[0,1]) and list(m.feature_names_in_)==list(X) and m.n_features_in_==21
  loaded=joblib.load(ckpt);q=V.iloc[:1024];p=m.predict_proba(q)[:,1];reloaded=loaded.predict_proba(q)[:,1];reload_error=float(np.max(np.abs(p-reloaded)));reconstruction_error=float(np.max(np.abs(p-reconstruct_probability(loaded,q))));batch_error=float(np.max(np.abs(p[:37]-loaded.predict_proba(q.iloc[:37])[:,1])));assert max(reload_error,reconstruction_error,batch_error)<1e-12
  del m,loaded;gc.collect();m=joblib.load(ckpt);prediction=np.full(len(vi),np.nan)
  for start in range(0,len(vi),16384):
   take=vi[start:start+16384];query=V.iloc[start:start+16384];file=run/f'query{start:06d}.npz'
   pv=m.predict_proba(query)[:,1];assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();save_npz(file,idx=take,id=ids[take],pred=pv,plan_sha256=np.asarray(ph));prediction[start:start+len(take)]=pv
   state('query_prediction',stage=stage,fold=k,predicted_rows=min(start+16384,len(vi)),valid_rows=len(vi),seconds=time.monotonic()-tick,peak_RSS_bytes=peak[0])
  save_npz(run/'validation.npz',idx=vi,id=ids[vi],target=y[vi],pred=prediction)
  assert sha(ckpt)==fit['model_sha256'];files={n:sha(run/n) for n in ['model.joblib','training.json','inner_bags.npz','outer_indices.npz','validation.npz']};rec=dict(status='complete',stage=stage,fold=k,plan_sha256=ph,runner_sha256=sha(__file__),files_sha256=files,model_reload_error=reload_error,additive_formula_error=reconstruction_error,query_batch_error=batch_error,peak_RSS_bytes=max(peak[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),seconds=time.monotonic()-tick,outer_quality_scored=False,GPU_used=False);dump(run/'result.json',rec);state('fold_complete',stage=stage,fold=k,seconds=rec['seconds'])
 finally:stop.set()
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

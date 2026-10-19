"""One frozen TabSTAR outerfit; explicit native inner split, then blind queries."""
from research_r79_model import ROOT,torch,np,parameters,configure,make_model,digest,predict,preprocessor_state,trace_native,TabSTARClassifier
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from tabstar.preprocessing.splits import split_to_val
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,os,json,sys,time,hashlib,threading,resource,gc,signal
from research_run_deadline import check_training,check_finalization,StopByDeadline
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);t=p.with_name('.'+p.name+f'.{os.getpid()}.{threading.get_native_id()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
lock=threading.RLock();progress={}
def state(status,**kw):
 with lock:
  progress.update(status=status,**kw);d=dict(updated_utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),outer_quality_scored=False,**progress);dump(ROOT/'research_r79_state.json',d);print(json.dumps(d),flush=True)
def npz(p,**a):
 if p.exists():
  z=np.load(p);assert set(z.files)==set(a) and all(np.array_equal(z[k],v) for k,v in a.items());return
 t=p.with_name('.'+p.stem+'.npz');np.savez_compressed(t,**a);os.replace(t,p)
def main(stage,k):
 check_finalization()
 plan=json.loads((ROOT/'research_r79_plan.json').read_text());ph=sha(ROOT/'research_r79_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r79_tabstar_raw21_fivefold/result.json').read_text())['eligible']
 run=ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'result.json').exists():
  rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==ph and all(sha(run/n)==h for n,h in rec['files_sha256'].items());return
 configure();assert torch.cuda.is_available() and host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 assert __import__('shutil').disk_usage(ROOT).free>2*2**30
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 def beat():
  while not stop.wait(30):
   with lock:state(progress.get('status','starting'),seconds=time.monotonic()-tick,CPU_seconds=time.process_time(),peak_GPU_bytes=peak[0])
 state('loading_own_data',stage=stage,fold=k);threading.Thread(target=beat,daemon=True).start()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');assert frame.shape==(699635,23) and frame.satisfaction.dtype==bool and frame.id.is_unique
  ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(ids,base['id']) and np.array_equal(y,base['target'])
  K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(y),-1,dtype='int8')
  for j,(_,ii) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[ii]=j
  assert np.array_equal(fold,base['fold']);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);labels=pd.Series(np.where(y,'True','False'),index=raw.index,name='satisfaction')
  X,I,Y,J=split_to_val(raw.iloc[fi],labels.iloc[fi],is_cls=True);ti=X.index.to_numpy();iv=I.index.to_numpy();assert len(iv)==1002 and len(np.intersect1d(ti,iv))==0 and np.array_equal(np.sort(np.r_[ti,iv]),fi) and not np.isin(np.r_[ti,iv],vi).any()
  npz(run/'indices.npz',fi=fi,vi=vi,inner_train=ti,inner_valid=iv,train_id=ids[fi],valid_id=ids[vi],inner_train_id=ids[ti],inner_valid_id=ids[iv]);del frame
  ckpt=run/'model.joblib';metadata=run/'training.json';reload_error=0.
  if metadata.exists():
   fit=json.loads(metadata.read_text());assert fit['plan_sha256']==ph and sha(ckpt)==fit['model_sha256'];model=TabSTARClassifier.load(str(ckpt));model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True
   state('restored_completed_fit',stage=stage,fold=k)
  else:
   check_training()
   model=make_model(run/'native_checkpoints');state('native_GPU_fit',stage=stage,fold=k,own_train_rows=len(ti),inner_valid_rows=len(iv),max_epochs=3)
   with trace_native(lambda status,**kw:state(status,stage=stage,fold=k,**kw),plan['pretrained_base_parameters_sha256']) as trace:model.fit(X,Y,x_val=I,y_val=J)
   assert len(trace['epochs'])==len(trace['inner_evaluations'])==3 and trace['updates_executed']>0 and digest(model.model_,True)==plan['pretrained_base_parameters_sha256']
   assert all(torch.isfinite(p).all() for p in model.model_.parameters());best_epoch=int(np.argmax([q['metric'] for q in trace['inner_evaluations']]))+1
   prep=preprocessor_state(model.preprocessor_);dump(run/'preprocessor.json',prep);assert prep['date_columns']==[] and prep['constant_columns']==[] and prep['target_classes']==['False','True']
   reference=predict(model,raw.iloc[vi[:64]]);model.model_.to('cpu');model.device=torch.device('cpu');model.use_amp=False;temp=run/'.model.joblib.tmp';model.save(str(temp));os.replace(temp,ckpt);final_parameter_hash=digest(model.model_)
   best=Path(trace['best_dir']);files=[p for p in best.iterdir() if p.is_file()];assert any(p.suffix=='.safetensors' for p in files)
   fit=dict(plan_sha256=ph,model_sha256=sha(ckpt),parameters=parameters(),own_outertrain_rows=len(fi),inner_train_rows=len(ti),inner_valid_rows=len(iv),positive_class='True',outer_validation_used_for_fit=False,pretrained_base_parameters_sha256=plan['pretrained_base_parameters_sha256'],final_parameters_sha256=final_parameter_hash,best_epoch=best_epoch,trace=trace,preprocessor_sha256=sha(run/'preprocessor.json'),native_best_checkpoint_files={str(p.relative_to(run)):sha(p) for p in files},fit_seconds=time.monotonic()-tick);dump(metadata,fit)
   del model;gc.collect();torch.cuda.empty_cache();model=TabSTARClassifier.load(str(ckpt));assert digest(model.model_)==final_parameter_hash;model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True;reload_error=float(np.max(np.abs(reference-predict(model,raw.iloc[vi[:64]]))));assert reload_error<1e-6
  assert digest(model.model_,True)==plan['pretrained_base_parameters_sha256'];prediction=np.full(len(vi),np.nan)
  for start in range(0,len(vi),16384):
   ix=vi[start:start+16384];file=run/f'query{start:06d}.npz'
   if file.exists():
    q=np.load(file);assert str(q['plan_sha256'])==ph and np.array_equal(q['idx'],ix) and np.array_equal(q['id'],ids[ix]);p=q['pred']
    assert np.max(np.abs(p[:64]-predict(model,raw.iloc[ix[:64]])))<1e-6
   else:p=predict(model,raw.iloc[ix]);npz(file,idx=ix,id=ids[ix],pred=p,plan_sha256=np.array(ph))
   prediction[start:start+len(ix)]=p;state('outer_query_prediction',stage=stage,fold=k,predicted_rows=min(start+16384,len(vi)),total_rows=len(vi),seconds=time.monotonic()-tick,peak_GPU_bytes=peak[0])
  npz(run/'validation.npz',idx=vi,id=ids[vi],target=y[vi],pred=prediction);assert np.isfinite(prediction).all() and sha(ckpt)==fit['model_sha256']
  first=raw.iloc[vi[:64]];batch_error=float(np.max(np.abs(predict(model,first)[:7]-predict(model,first.iloc[:7]))));assert batch_error<.002 and digest(model.model_)==fit['final_parameters_sha256']
  files={n:sha(run/n) for n in ['model.joblib','training.json','preprocessor.json','indices.npz','validation.npz']};rec=dict(status='complete',stage=stage,fold=k,plan_sha256=ph,runner_sha256=sha(__file__),files_sha256=files,reload_error=reload_error,batch_error=batch_error,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,outer_quality_scored=False);dump(run/'result.json',rec);state('fold_complete',stage=stage,fold=k,seconds=rec['seconds'])
 except BaseException as error:
  dump(run/'interrupted.json',dict(error_type=type(error).__name__,error=str(error)[:600],plan_sha256=ph,progress=progress,native_trace=locals().get('trace'),no_candidate_qualification=True));raise
 finally:stop.set()
if __name__=='__main__':
 def terminate(signum,frame):raise StopByDeadline('NormalSIGTERMfromtaskcontrolleroruserdeadline')
 signal.signal(signal.SIGTERM,terminate)
 try:main(sys.argv[1],int(sys.argv[2]))
 except StopByDeadline as error:state('stopped_by_user_deadline',reason=str(error));sys.exit(0)

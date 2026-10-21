"""Prospective inner-only TabSTAR compute control. Reuse3epoch, one12epoch fit."""
import os
os.environ['TQDM_DISABLE']='1'
from research_r79r2_model import ROOT,torch,np,parameters,configure,digest,predict,preprocessor_state,trace_native,TabSTARClassifier,TabStarTrainer
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from research_resume_policy_20261021 import check_training,StopByDeadline
from tabstar.preprocessing.splits import split_to_val
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,threading,resource,gc,shutil,signal,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,d):
 p=Path(p);t=p.with_name('.'+p.name+f'.{os.getpid()}.{threading.get_native_id()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
lock=threading.RLock();progress={};live_trace={}
def state(status,**kw):
 with lock:
  progress.update(status=status,**kw);d=dict(updated_utc=now(),PID=os.getpid(),outer_quality_scored=False,**progress);dump(ROOT/'research_r80_state.json',d);print(json.dumps(d),flush=True)
def metrics(y,p):return dict(AUC=float(roc_auc_score(y,p)),logloss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])))
def main():
 check_training();plan=json.loads((ROOT/'research_r80_plan.json').read_text());ph=sha(ROOT/'research_r80_plan.json');out=ROOT/'research_artifacts/r80_tabstar_3v12_inner_control';out.mkdir(exist_ok=True)
 assert not (out/'result.json').exists() and not (out/'started.json').exists(),'Existingpilotmustbeassessed;noautomaticduplicate'
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 configure();assert torch.cuda.is_available() and host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and shutil.disk_usage(ROOT).free>=int(2.5*2**30);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 dump(out/'started.json',dict(started_utc=now(),plan_sha256=ph,PID=os.getpid()));stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 def monitor():
  while not stop.wait(30):
   free=shutil.disk_usage(ROOT).free;own=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
   if free<2*2**30 or own>512*2**20:dump(out/'disk_failure.json',dict(free_bytes=free,own_output_bytes=own));os._exit(86)
   state(progress.get('status','starting'),seconds=time.monotonic()-tick,CPU_seconds=time.process_time(),peak_GPU_bytes=peak[0],disk_free_bytes=free,own_output_bytes=own,trace_snapshot=live_trace)
 state('verifying_same_own_inner_split',plan_sha256=ph,max_epochs=12);threading.Thread(target=monitor,daemon=True).start()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');assert frame.shape==(699635,23) and frame.satisfaction.dtype==bool;ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);fi,vi=next(StratifiedKFold(5,shuffle=True,random_state=20261018).split(raw,y));old=ROOT/'research_artifacts/r79r2_tabstar_raw21_fivefold/fold0';ix=np.load(old/'indices.npz');assert np.array_equal(ix['fi'],fi) and np.array_equal(ix['vi'],vi);labels=pd.Series(np.where(y,'True','False'),index=raw.index,name='satisfaction');X,I,Y,J=split_to_val(raw.iloc[fi],labels.iloc[fi],is_cls=True);ti=X.index.to_numpy();iv=I.index.to_numpy();assert np.array_equal(ti,ix['inner_train']) and np.array_equal(iv,ix['inner_valid']) and len(iv)==1002 and not np.isin(np.r_[ti,iv],vi).any();np.savez_compressed(out/'indices.npz',fi=fi,vi=vi,inner_train=ti,inner_valid=iv,inner_train_id=ids[ti],inner_valid_id=ids[iv]);oldfit=json.loads((old/'training.json').read_text());oldbest=oldfit['best_epoch']-1;assert oldfit['parameters']==parameters() and len(oldfit['trace']['epochs'])==3
  # Replay only the old checkpoint's own inner rows. No completed3epoch fit is repeated.
  state('replaying_existing_3epoch_inner_checkpoint',inner_train_rows=len(ti),inner_valid_rows=len(iv));model=TabSTARClassifier.load(str(old/'model.joblib'));assert digest(model.model_)==oldfit['final_parameters_sha256'];model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True;p3=predict(model,I);m3=metrics(y[iv],p3);assert abs(m3['AUC']-oldfit['trace']['inner_evaluations'][oldbest]['metric'])<1e-4;np.savez_compressed(out/'existing3_inner.npz',idx=iv,id=ids[iv],target=y[iv],pred=p3);dump(out/'existing3.json',dict(source_training_sha256=sha(old/'training.json'),source_model_sha256=sha(old/'model.joblib'),native_best_epoch=oldbest+1,native_inner=oldfit['trace']['inner_evaluations'][oldbest],replayed_inner=m3,trace=oldfit['trace'],no_refit=True));del model;gc.collect();torch.cuda.empty_cache()
  pars=plan['parameters_12'];assert {k:v for k,v in pars.items() if k not in ['max_epochs','patience']}=={k:v for k,v in parameters().items() if k not in ['max_epochs','patience']};model=TabSTARClassifier(device=torch.device('cuda'),output_dir=str(out/'native_checkpoints'),**pars);state('native_GPU_12epoch_fit',inner_train_rows=len(ti),inner_valid_rows=len(iv),max_epochs=12)
  def callback(status,**kw):state(status,**kw)
  with trace_native(callback,plan['pretrained_base_parameters_sha256']) as trace:
   live_trace.update(trace)
   # Share native list observations, without replacing optimizer/scheduler/math.
   model.fit(X,Y,x_val=I,y_val=J)
  live_trace.update(trace);assert len(trace['epochs'])==len(trace['inner_evaluations'])==12 and trace['updates_executed']>0;assert digest(model.model_,True)==plan['pretrained_base_parameters_sha256'];assert all(torch.isfinite(p).all() for p in model.model_.parameters());assert all(abs(e['optimizer_groups'][0]['lr']-lr)<1e-15 for e,lr in zip(trace['epochs'],plan['epoch_start_LRs']['12']));best=int(np.argmax([q['metric'] for q in trace['inner_evaluations']]));prep=preprocessor_state(model.preprocessor_);assert prep==json.loads((old/'preprocessor.json').read_text());dump(out/'preprocessor.json',prep);p12=predict(model,I);m12=metrics(y[iv],p12);assert abs(m12['AUC']-trace['inner_evaluations'][best]['metric'])<1e-4;np.savez_compressed(out/'new12_inner.npz',idx=iv,id=ids[iv],target=y[iv],pred=p12)
  state('serializing_selected_inner_checkpoint',best_epoch=best+1);model.model_.to('cpu');model.device=torch.device('cpu');model.use_amp=False;temp=out/'.model.joblib.tmp';model.save(str(temp));os.replace(temp,out/'model.joblib');final_hash=digest(model.model_);bestdir=Path(trace['best_dir']);checkpoint_files={str(p.relative_to(out)):sha(p) for p in bestdir.iterdir() if p.is_file()};dump(out/'training.json',dict(plan_sha256=ph,parameters=pars,pretrained_base_parameters_sha256=plan['pretrained_base_parameters_sha256'],final_parameters_sha256=final_hash,model_sha256=sha(out/'model.joblib'),best_epoch=best+1,trace=trace,preprocessor_exact_vs_3epoch=True,outer_validation_used=False,fit_seconds=time.monotonic()-tick,native_best_checkpoint_files=checkpoint_files));del model;gc.collect();torch.cuda.empty_cache();model=TabSTARClassifier.load(str(out/'model.joblib'));assert digest(model.model_)==final_hash;model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True;reload_error=float(np.max(np.abs(p12-predict(model,I))));assert reload_error<1e-6
  native3=oldfit['trace']['inner_evaluations'][oldbest];native12=trace['inner_evaluations'][best];worth=bool(best+1>3 and native12['metric']>native3['metric'] and native12['loss']<native3['loss']);from compare_validation import compare
  result=dict(status='complete_inner_control',completed_utc=now(),plan_sha256=ph,completed_new_fits=1,reused_complete_3epoch_fit=True,epochs_completed=12,existing3_native_best_epoch=oldbest+1,new12_native_best_epoch=best+1,existing3_native_inner=native3,new12_native_inner=native12,inner_selected_AUC_delta=native12['metric']-native3['metric'],inner_selected_loss_delta=native12['loss']-native3['loss'],inner_prediction_comparison_descriptive_only=compare(y[iv],p3,p12),worth_prospective_outer_validation=worth,mechanism='Native12epochhorizonand3epochhorizonhavedifferentLRs;testscombinedtrainingbudget/schedulerrecipe,notpureepochcausalattribution',outer_quality_scored=False,threefold_started=False,CSV_created=False,formal_submissions=0,model_reload_error=reload_error,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,files_sha256={n:sha(out/n) for n in ['model.joblib','training.json','preprocessor.json','indices.npz','existing3.json','existing3_inner.npz','new12_inner.npz']},statistics_limitation=plan['statistics_limitation']);dump(out/'result.json',result);dump(ROOT/'research_r80_result.json',result);state('complete_inner_control',worth_prospective_outer_validation=worth,best_epoch=best+1,seconds=result['seconds']);print(json.dumps(result,indent=2),flush=True)
 except BaseException as e:dump(out/'interrupted.json',dict(error_type=type(e).__name__,error=str(e)[:800],plan_sha256=ph,trace=live_trace,progress=progress));state('interrupted',error_type=type(e).__name__,error=str(e)[:300]);raise
 finally:stop.set()
if __name__=='__main__':
 def terminate(signum,frame):raise StopByDeadline('Explicitstopforthispilotonly')
 signal.signal(signal.SIGTERM,terminate);main()

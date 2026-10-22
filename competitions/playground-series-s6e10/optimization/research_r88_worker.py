"""One frozen outerfit per process. No held-out labels in fit or epoch selection."""
from research_r88_model import *
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from research_w3_common import dump,now
from research_resume_policy_20261021 import check_training
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
import pandas as pd,joblib,cloudpickle,gc,threading,resource,time,sys,shutil
def main(stage,k):
 check_training();plan=json.loads((ROOT/'research_r88_plan.json').read_text());ph=sha(ROOT/'research_r88_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 assert plan['config']==config() and json.loads((ROOT/'research_r88_probe.json').read_text())['status']=='passed_no_quality_score'
 if stage=='threefold':assert json.loads((ROOT/'research_r88_result.json').read_text())['stages']['fivefold']['passed']
 out=ROOT/'research_artifacts/r88_missing_category_nn'/stage/f'fold{k}';out.mkdir(parents=True,exist_ok=True);record=out/'result.json'
 if record.exists():
  done=json.loads(record.read_text())
  if done['status']=='complete':
   assert done['plan_sha256']==ph and all(sha(out/n)==h for n,h in done['files_sha256'].items());print('SKIP_VERIFIED_COMPLETE',stage,k,flush=True);return
  raise RuntimeError('Partialfoldrequiresassessment;noautomaticretraining')
 assert shutil.disk_usage(ROOT).free>=2*2**30 and host_available()>=6*2**30
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');y=frame.satisfaction.to_numpy('int8');ids=frame.id.to_numpy();raw=frame.drop(columns=['id','satisfaction']);base,tree=references(stage);K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);splits=list(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y));fi,vi=splits[k]
  assert np.array_equal(ids,base['id']) and np.array_equal(y,base['target']) and np.array_equal(vi,np.flatnonzero(base['fold']==k));a=auxiliary(ids);tr=MissingCategoryNeuralFeatures()
  with threadpool_limits(6):X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],a.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),a.iloc[vi].reset_index(drop=True))
  control=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[vi].reset_index(drop=True),a.iloc[vi].reset_index(drop=True));column=tr.repaired_column;mask=raw.iloc[vi]['Arrival Delay in Minutes'].isna().to_numpy();C.loc[mask,column]='__MISSING__';pd.testing.assert_frame_equal(V,C,check_exact=True);del C,control
  assert X.shape==(len(fi),74) and X.select_dtypes(include='number').shape[1]==52;joblib.dump(tr,out/'transformer.joblib');np.savez_compressed(out/'indices.npz',fi=fi,vi=vi,train_id=ids[fi],valid_id=ids[vi])
  # Save strict cross-fitted train encodings for independent leakage/replay audit.
  np.savez_compressed(out/'training_te.npz',idx=fi,values=X[[f'cross_te_{j}' for j in range(3)]].to_numpy())
  rec=dict(status='training',started_utc=now(),PID=os.getpid(),stage=stage,fold=k,plan_sha256=ph,config=config(),train_rows=len(fi),valid_rows=len(vi),feature_width=74,original73_and_nonmissing_category_exact=True,missing_category_column=column,validation_missing_rows=int(mask.sum()),validation_used_for_fit_or_epoch_selection=False,epochs=[]);dump(record,rec);print('START',stage,k,rec['started_utc'],flush=True);torch.cuda.reset_peak_memory_stats();previous=[time.monotonic()];original=StreamingTabNNModule.on_train_epoch_end;original_step=StreamingTabNNModule.training_step
  def step_hook(module,batch,batch_idx):
   if batch_idx==0:assert sorted({str(v.device) for v in batch.values()})==['cuda:0'];rec['observed_batch_devices']=['cuda:0']
   return original_step(module,batch,batch_idx)
  def hook(module):
   original(module);torch.cuda.synchronize();check_training();devices=sorted({str(p.device) for p in module.parameters()});assert devices==['cuda:0'] and str(module.train_dl.ds.device)=='cpu';t=time.monotonic();rec['epochs'].append(dict(epoch=int(module.current_epoch)+1,seconds=t-previous[0],training_parameter_devices=devices,backing_data_device='cpu',peak_GPU_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated()));previous[0]=t;dump(record,rec);print('EPOCH',stage,k,rec['epochs'][-1],flush=True)
  StreamingTabNNModule.on_train_epoch_end=hook;StreamingTabNNModule.training_step=step_hook
  try:
   with streaming_backend(),threadpool_limits(6):model=BoundedRealMLP(**config()).fit(X,y[fi])
  finally:StreamingTabNNModule.on_train_epoch_end=original;StreamingTabNNModule.training_step=original_step
  assert len(rec['epochs'])==3 and not model.cv_alg_interface_.model.creator.is_cv and np.array_equal(model.classes_,[0,1]);pred=model.predict_proba(V)[:,1];assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();model.to('cpu')
  with (out/'model.pkl').open('wb') as f:cloudpickle.dump(model,f)
  del model,X;gc.collect();torch.cuda.empty_cache()
  with (out/'model.pkl').open('rb') as f:restored=cloudpickle.load(f)
  restored.to('cuda:0');error=float(np.max(np.abs(restored.predict_proba(V.iloc[:2048])[:,1]-pred[:2048])));assert error<1e-7
  np.savez_compressed(out/'validation.npz',idx=vi,id=ids[vi],target=y[vi],pred=pred);rec.update(status='complete',ended_utc=now(),elapsed_seconds=time.monotonic()-tick,model_reload_error=error,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,files_sha256={n:sha(out/n) for n in ['transformer.joblib','model.pkl','indices.npz','training_te.npz','validation.npz']});dump(record,rec);print('DONE',stage,k,rec['elapsed_seconds'],flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

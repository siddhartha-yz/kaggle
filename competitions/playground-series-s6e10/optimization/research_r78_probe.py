"""No SAT values or quality scores: fullshape matched32 GPU capacity check."""
from research_r78_model import ROOT,config,torch,np,LogratioNeuralFeatures,auxiliary,BoundedRealMLP,StreamingTabNNModule,streaming_backend
from research_r23_features import ConditionalFeatures
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from threadpoolctl import threadpool_limits
from datetime import datetime,timezone
import pandas as pd,json,time,threading,resource,gc,cloudpickle,hashlib
def main():
 assert json.loads((ROOT/'research_r79r2_result.json').read_text())['status']=='not_qualified'
 torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'research_artifacts/r78_resource_probe';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv',usecols=lambda c:c!='satisfaction');ids=frame.pop('id').to_numpy();a=auxiliary(ids);fold=np.load(ROOT/'research_artifacts/r23_conditional_lean_fivefold/oof.npz')['fold'];fi=np.flatnonzero(fold!=0);vi=np.flatnonzero(fold==0);y=np.random.default_rng(7801).integers(0,2,len(fi),dtype='int8')
  # CPU-only metamorphic check before any neural fit.
  n=4096;new=LogratioNeuralFeatures();old=ConditionalFeatures();small=frame.iloc[fi[:n]].reset_index(drop=True);v=a.iloc[fi[:n]].reset_index(drop=True);A=new.fit_transform(small,y[:n],v);B=old.fit_transform(small,y[:n],v.iloc[:,:27]);assert A.iloc[:,:74].equals(B)
  tr=LogratioNeuralFeatures();X=tr.fit_transform(frame.iloc[fi].reset_index(drop=True),y,a.iloc[fi].reset_index(drop=True));V=tr.transform(frame.iloc[vi[:4096]].reset_index(drop=True),a.iloc[vi[:4096]].reset_index(drop=True));assert X.shape==(559708,139) and X.select_dtypes(include='number').shape[1]==117
  epochs=[];original=StreamingTabNNModule.on_train_epoch_end
  def hook(module):
   original(module);torch.cuda.synchronize();assert all(p.device.type=='cuda' for p in module.parameters()) and str(module.train_dl.ds.device)=='cpu';epochs.append(dict(epoch=int(module.current_epoch)+1,elapsed_seconds=time.monotonic()-tick,peak_GPU_bytes=peak[0]))
  StreamingTabNNModule.on_train_epoch_end=hook
  try:
   with streaming_backend(),threadpool_limits(6):model=BoundedRealMLP(**config(1)).fit(X,y)
  finally:StreamingTabNNModule.on_train_epoch_end=original
  assert len(epochs)==1 and np.array_equal(model.classes_,[0,1]);p=model.predict_proba(V)[:,1];assert np.isfinite(p).all();model.to('cpu');cpu=model.predict_proba(V.iloc[:16])[:,1];assert np.isfinite(cpu).all();cpu_error=float(np.max(np.abs(cpu-p[:16])));assert cpu_error<1e-4
  path=out/'synthetic_model.pkl'
  with path.open('wb') as f:cloudpickle.dump(model,f)
  del model;gc.collect();torch.cuda.empty_cache()
  with path.open('rb') as f:restored=cloudpickle.load(f)
  restored.to('cuda:0');reload_error=float(np.max(np.abs(p[:2048]-restored.predict_proba(V.iloc[:2048])[:,1])));assert reload_error<1e-7
  result=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),train_rows=len(fi),features=139,numeric_features=117,nominal_features=22,native_members=32,batch_size=256,CPU_threads=6,epochs=epochs,original_74_CPU_inputs_exact=True,official_SAT_values_loaded=False,classifier_labels='Syntheticseed7801;notSAT',outer_quality_scored=False,reload_error=reload_error,CPU_GPU_prediction_error=cpu_error,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,config=config(1),model_not_accepted_for_formal_validation=True)
  (ROOT/'research_r78_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

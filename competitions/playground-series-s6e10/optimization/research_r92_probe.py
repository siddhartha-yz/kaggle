"""No quality scoring: saved native outputs, replay/restoration, throughput and invariance."""
from research_r92_gpu import *
def main():
 p=plan();run=OUT/'probe';assert not (run/'result.json').exists();stop,peak,monitor=setup(run);tick=time.monotonic()
 try:
  src,frame,raw,y,r,fi,vi,a,tr,X,z,reconstruct_seconds=inputs('fivefold',0,p);Q=X.iloc[:2048]
  model=load_model(src,'cpu');module=model.cv_alg_interface_.model
  with threadpool_limits(6):cpu=predict_margins(model,Q.iloc[:256])
  model.to('cuda:0');module.creator.device_info='cuda:0';torch.cuda.reset_peak_memory_stats()
  with threadpool_limits(6):
   default=model.predict_proba(Q)[:,1]
   with native_logits(model):
    raw_native=model._predict_raw(Q);assert list(raw_native.shape)==[32,2048,2];arr=raw_native.numpy().astype('float64');direct=arr[:,:,1]-arr[:,:,0]
   begin=time.monotonic();native=predict_margins(model,Q);native_seconds=time.monotonic()-begin
   again=model.predict_proba(Q)[:,1]
   try:
    with native_logits(model):raise RuntimeError('R92 deliberate restoration fixture')
   except RuntimeError as e:assert str(e)=='R92 deliberate restoration fixture'
   after_exception=model.predict_proba(Q)[:,1]
  replay=float(np.max(np.abs(default-z['raw_pred'][:2048])));meanerr=float(np.max(np.abs(expit(native).mean(0)-default)));restored=float(np.max(np.abs(default-again)));exception_restore=float(np.max(np.abs(default-after_exception)));cpuerr=float(np.max(np.abs(cpu-native[:,:256])));nativeerr=float(np.max(np.abs(native-direct)))
  assert replay<=2e-6 and meanerr<=2e-6 and restored<=1e-7 and exception_restore<=1e-7 and cpuerr<=2e-5 and nativeerr<=1e-7
  assert '_postprocess_ens_pred' not in vars(module)
  # Synthetic-only fit of the fixed origin moment equations; no official calibration or quality.
  rng=np.random.default_rng(20261070);ty=np.r_[np.zeros(400,dtype='int8'),np.ones(600,dtype='int8')];m=rng.normal(size=(32,1000));v=rng.normal(size=(32,200));shift=rng.normal(0,10,size=32);b=fit_origins(m,ty);bs=fit_origins(m+shift[:,None],ty)
  invariant=float(np.max(np.abs(predict_calibrated(v,b)-predict_calibrated(v+shift[:,None],bs))));assert invariant<1e-12
  source('fivefold',0,p)
  estimate=float(2*native_seconds/len(Q)*len(frame)*5+reconstruct_seconds*5+150)
  record=dict(status='passed_no_quality_score',completed_utc=now(),plan_sha256=sha(ROOT/'research_r92_plan.json'),native_raw_shape=list(raw_native.shape),training_queries_only=True,official_validation_quality_computed=False,new_classifier_fits=0,official_calibration_fits=0,CPU_GPU_native_margin_error=cpuerr,current_raw_probability_replay_error=replay,native_probability_mean_error=meanerr,native_helper_error=nativeerr,normal_restore_error=restored,exception_restore_error=exception_restore,synthetic_origin_invariance_error=invariant,source_files_unchanged=True,native_2048_seconds=native_seconds,training_frame_reconstruction_seconds=reconstruct_seconds,estimated_fivefold_seconds=estimate,estimation='2x warmednativequeryrateplusreconstructionand150seccalibration/IOallowance; checkpointnotcampaignstop',peak_GPU_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick)
  assert 0<peak[0]<=p['resources']['GPU_process_ceiling_bytes'] and record['peak_RSS_bytes']<=8*2**30
  write_guard();dump(run/'result.json',record);dump(ROOT/'research_r92_probe_result.json',record);print('PROBE_PASSED',json.dumps(record),flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

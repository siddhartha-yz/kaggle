"""Saved weights only: fit32 own-training origins, then infer untouched outer queries."""
from research_r92_gpu import *
import sys,hashlib

def main(stage,k):
 p=plan();probe=json.loads((OUT/'probe/result.json').read_text());assert probe['status']=='passed_no_quality_score' and probe['plan_sha256']==sha(ROOT/'research_r92_plan.json')
 if stage=='threefold':assert json.loads((OUT/'fivefold/result.json').read_text())['eligible']
 run=OUT/stage/f'fold{k}';write_guard(12)
 if (run/'result.json').exists():
  r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r92_plan.json')
  for n,h in r['files_sha256'].items():assert sha(run/n)==h
  print('REUSED_VERIFIED',stage,k,flush=True);return
 stop,peak,monitor=setup(run);tick=time.monotonic()
 try:
  rec=dict(status='preparing_own_training_inputs',stage=stage,fold=k,pid=os.getpid(),started_utc=now(),plan_sha256=sha(ROOT/'research_r92_plan.json'),new_neural_fits=0,official_validation_labels_used_for_calibration=False)
  dump(run/'result.json',rec);print('START_ORIGIN_CALIBRATION',stage,k,rec['pid'],flush=True)
  src,frame,raw,y,r,fi,vi,a,tr,X,z,reconstruct_seconds=inputs(stage,k,p);model=load_model(src,'cuda:0');torch.cuda.reset_peak_memory_stats()
  with threadpool_limits(6):M,training_seconds=margins_with_progress(model,X,'own_training_native_inference',run/'progress.json',stage,k)
  reconstructed_raw=expit(M).mean(0);train_replay=float(np.max(np.abs(reconstructed_raw-z['raw_pred'])));assert train_replay<=2e-6
  begin=time.monotonic();bias=fit_origins(M,y[fi]);calibration_seconds=time.monotonic()-begin;prior=float(y[fi].mean());moments=expit(M+bias[:,None]).mean(1);moment_error=float(np.max(np.abs(moments-prior)));assert moment_error<=1e-10
  shift=np.linspace(-4.,4.,32);invariance=float(np.max(np.abs(predict_calibrated(M[:,:256],bias)-predict_calibrated(M[:,:256]+shift[:,None],bias-shift))));assert invariance<1e-12
  old_cal=json.loads((src/'calibration.json').read_text());newtrain=predict_calibrated(M,bias);oldtrain=calibrate(z['raw_pred'],old_cal['bias']);train_operator_change=float(np.max(np.abs(newtrain-oldtrain)))
  origins=dict(kind='32 additive score origins before uniformprobabilitymean; noslopes/newweights',bias=bias.tolist(),training_rows=len(fi),training_prior=prior,training_indices_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),source_model_sha256=p['models'][stage][k]['files_sha256']['model.pkl'],moment_error=moment_error,source_global_bias_replaced=old_cal['bias'],in_sample_calibration=True,validation_labels_used=False)
  write_guard();dump(run/'origins.json',origins)
  np.savez_compressed(run/'training_fixture.npz',idx=fi[:256],id=frame.id.to_numpy()[fi[:256]],margin=M[:,:256],old_raw=z['raw_pred'][:256],new_probability=newtrain[:256])
  del M,X,newtrain,oldtrain,reconstructed_raw;gc.collect()
  V=tr.transform(raw.iloc[vi].reset_index(drop=True),a.iloc[vi].reset_index(drop=True));assert V.shape==(len(vi),74)
  with threadpool_limits(6):MV,validation_seconds=margins_with_progress(model,V,'outer_native_inference',run/'progress.json',stage,k)
  q=predict_calibrated(MV,bias);legacy=calibrate(expit(MV).mean(0),old_cal['bias']);validation_replay=float(np.max(np.abs(legacy-r['pred'][vi])));assert validation_replay<=2e-6
  assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all() and '_postprocess_ens_pred' not in vars(model.cv_alg_interface_.model)
  write_guard(12);np.savez_compressed(run/'validation.npz',idx=vi,id=frame.id.to_numpy()[vi],pred=q,legacy_replay=legacy)
  np.savez_compressed(run/'validation_fixture.npz',idx=vi[:256],margin=MV[:,:256],new_probability=q[:256],legacy_replay=legacy[:256])
  source(stage,k,p)
  rec.update(status='complete',completed_utc=now(),train_rows=len(fi),valid_rows=len(vi),source_model_sha256=origins['source_model_sha256'],native_members=32,train_index_sha256=origins['training_indices_sha256'],origins_count=32,slope_fits=0,weight_fits=0,own_training_input_reconstruction_seconds=reconstruct_seconds,training_inference_seconds=training_seconds,calibration_seconds=calibration_seconds,validation_inference_seconds=validation_seconds,own_training_raw_replay_error=train_replay,outer_legacy_replay_error=validation_replay,training_moment_error=moment_error,origin_invariance_error=invariance,own_training_operator_change_max=train_operator_change,bias_range=[float(bias.min()),float(bias.max())],peak_GPU_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,files_sha256={n:sha(run/n) for n in ['origins.json','training_fixture.npz','validation.npz','validation_fixture.npz']})
  assert peak[0]<=p['resources']['GPU_process_ceiling_bytes'] and rec['peak_RSS_bytes']<=8*2**30;dump(run/'result.json',rec);print('DONE_ORIGIN_CALIBRATION',stage,k,rec['seconds'],flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

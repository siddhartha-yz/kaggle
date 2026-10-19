"""Conditional test inference only; no refit, both gates/audit/deadline required."""
from research_r79_model import ROOT,configure,predict,torch,np,TabSTARClassifier
from research_run_deadline import check_finalization,utc_seconds,HARD,StopByDeadline
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from pathlib import Path
from datetime import datetime,timezone
from scipy.special import expit,logit
import json,hashlib,gc,threading,resource,time,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 check_finalization();assert utc_seconds()+1500<HARD-120,'Need25minutesforinference,replay,handoff';fp=ROOT/'research_r79_final_plan.json';plan=json.loads(fp.read_text())
 for n,h in plan['required_sha256'].items():assert sha(ROOT/n)==h,n
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/result.json').read_text())['eligible']
 audit=json.loads((ROOT/'research_r79_independent_audit.json').read_text());assert audit['status']=='passed' and audit['plan_sha256']==plan['validation_plan_sha256']
 configure();assert host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);out=ROOT/'research_artifacts/r79_final_inference';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  raw=pd.read_csv(ROOT/'data/test.csv');ids=raw.pop('id').to_numpy();sample=pd.read_csv(ROOT/'data/sample_submission.csv');base=pd.read_csv(ROOT/plan['baseline_CSV'],float_precision='round_trip');assert raw.shape==(299844,21) and 'satisfaction' not in raw and np.array_equal(ids,sample.id) and np.array_equal(ids,base.id) and list(base)==list(sample)==['id','satisfaction'];values=[];proof=[]
  for k in range(5):
   check_finalization();run=ROOT/f'research_artifacts/r79_tabstar_raw21_fivefold/fold{k}';fit=json.loads((run/'training.json').read_text());assert sha(run/'model.joblib')==fit['model_sha256'] and fit['plan_sha256']==plan['validation_plan_sha256'];saved=out/f'fold{k}.npz'
   if saved.exists():
    z=np.load(saved);assert str(z['final_plan_sha256'])==sha(fp) and str(z['model_sha256'])==fit['model_sha256'] and np.array_equal(z['id'],ids);p=z['pred']
   else:
    model=TabSTARClassifier.load(str(run/'model.joblib'));model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True;parts=[]
    for start in range(0,len(ids),16384):
     check_finalization();parts.append(predict(model,raw.iloc[start:start+16384]));print(json.dumps(dict(status='qualified_test_prediction',fold=k,rows=min(start+16384,len(ids)),total=len(ids),updated_utc=datetime.now(timezone.utc).isoformat())),flush=True)
    p=np.concatenate(parts);np.savez_compressed(saved,id=ids,pred=p,model_sha256=np.array(fit['model_sha256']),final_plan_sha256=np.array(sha(fp)));del model;gc.collect();torch.cuda.empty_cache()
   assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();values.append(p);proof.append(dict(fold=k,model_sha256=fit['model_sha256'],prediction_sha256=sha(saved)))
  mean=np.mean(np.stack(values),axis=0);prediction=expit(.9375*logit(np.clip(base.satisfaction.to_numpy(),1e-6,1-1e-6))+.0625*logit(np.clip(mean,1e-6,1-1e-6)));target=ROOT/plan['output_CSV'];tmp=target.with_suffix('.tmp');pd.DataFrame({'id':ids,'satisfaction':prediction}).to_csv(tmp,index=False,float_format='%.17g');tmp.replace(target)
  rec=dict(status='generated_requires_independent_final_replay',checked_utc=datetime.now(timezone.utc).isoformat(),final_plan_sha256=sha(fp),CSV=target.name,CSV_sha256=sha(target),CSV_bytes=target.stat().st_size,rows=len(ids),folds=proof,seconds=time.monotonic()-tick,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_classifier_fits=0,formal_submission=False,actual_top500_verified=False);(ROOT/'research_r79_final_inference.json').write_text(json.dumps(rec,indent=2)+'\n')
 finally:stop.set()
if __name__=='__main__':main()

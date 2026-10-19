"""Conditional EBM inference only: requires both gates and independent fit audit."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,threading,resource
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from research_r77_probe import watch
ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'research_r77_final_inference_plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 plan=json.loads(PLAN.read_text())
 for n,h in plan['required_sha256'].items():assert sha(ROOT/n)==h,n
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r77_ebm_raw21_{stage}/result.json').read_text())['eligible']
 audit=json.loads((ROOT/'research_r77_independent_audit.json').read_text());assert audit['status']=='passed' and audit['plan_sha256']==plan['validation_plan_sha256']
 out=ROOT/'research_artifacts/r77_final_inference';out.mkdir(exist_ok=True)
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start()
 try:
  raw=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');baseline=pd.read_csv(ROOT/plan['baseline_CSV'],float_precision='round_trip');assert raw.shape==(299844,22) and list(raw)[0]=='id' and np.array_equal(raw.id,sample.id) and np.array_equal(raw.id,baseline.id) and list(sample)==['id','satisfaction'] and list(baseline)==list(sample)
  ids=raw.pop('id').to_numpy();assert 'satisfaction' not in raw and raw.shape[1]==21
  probabilities=[];records=[]
  for k in range(5):
   run=ROOT/f'research_artifacts/r77_ebm_raw21_fivefold/fold{k}';fit=json.loads((run/'training.json').read_text());assert fit['plan_sha256']==plan['validation_plan_sha256'] and sha(run/'model.joblib')==fit['model_sha256'];model=joblib.load(run/'model.joblib');assert np.array_equal(model.classes_,[0,1]) and list(model.feature_names_in_)==list(raw)
   p=np.concatenate([model.predict_proba(raw.iloc[j:j+16384])[:,1] for j in range(0,len(ids),16384)]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   saved=out/f'fold{k}.npz';np.savez_compressed(saved,id=ids,pred=p,model_sha256=np.array(fit['model_sha256']),final_plan_sha256=np.array(sha(PLAN)))
   probabilities.append(p);records.append(dict(fold=k,model_sha256=fit['model_sha256'],prediction_sha256=sha(saved)));del model
  mean=np.mean(np.stack(probabilities),axis=0);v12=baseline.satisfaction.to_numpy();assert np.isfinite(v12).all() and ((v12>=0)&(v12<=1)).all()
  prediction=expit(.9375*logit(np.clip(v12,1e-6,1-1e-6))+.0625*logit(np.clip(mean,1e-6,1-1e-6)));assert np.isfinite(prediction).all()
  frame=pd.DataFrame({'id':ids,'satisfaction':prediction});target=ROOT/plan['output_CSV'];tmp=target.with_suffix('.tmp');frame.to_csv(tmp,index=False,float_format='%.17g');tmp.replace(target)
  write(ROOT/'research_r77_final_inference.json',dict(status='generated_requires_independent_final_replay',checked_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),rows=len(ids),CSV=target.name,CSV_sha256=sha(target),CSV_bytes=target.stat().st_size,folds=records,peak_RSS_bytes=max(peak[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),new_classifier_fits=0,formal_submission=False,actual_top500_verified=False))
 finally:stop.set()
if __name__=='__main__':main()

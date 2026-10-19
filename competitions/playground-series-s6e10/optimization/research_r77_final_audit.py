"""Independent full-test manual-term replay of the already qualified EBM CSV."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,threading,resource,io
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from research_r77_audit import manual_probability
from research_r77_probe import watch
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 planpath=ROOT/'research_r77_final_inference_plan.json';plan=json.loads(planpath.read_text())
 for n,h in plan['required_sha256'].items():assert sha(ROOT/n)==h,n
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r77_ebm_raw21_{stage}/result.json').read_text())['eligible']
 assert json.loads((ROOT/'research_r77_independent_audit.json').read_text())['status']=='passed'
 generated=json.loads((ROOT/'research_r77_final_inference.json').read_text());assert generated['status']=='generated_requires_independent_final_replay' and generated['plan_sha256']==sha(planpath)
 stop=threading.Event();peak=[0];out=ROOT/'research_artifacts/r77_final_inference';threading.Thread(target=watch,args=(stop,peak,out/'audit_resource_failure.json'),daemon=True).start()
 try:
  raw=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');baseline=pd.read_csv(ROOT/plan['baseline_CSV'],float_precision='round_trip');csvpath=ROOT/plan['output_CSV'];csv=pd.read_csv(csvpath,float_precision='round_trip');ids=raw.pop('id').to_numpy();assert len(ids)==299844 and list(csv)==list(sample)==['id','satisfaction'] and all(np.array_equal(ids,f.id.to_numpy()) for f in [csv,baseline,sample]);assert 'satisfaction' not in raw and raw.shape[1]==21 and sha(csvpath)==generated['CSV_sha256']
  rebuilt=[];proofs=[]
  for k in range(5):
   run=ROOT/f'research_artifacts/r77_ebm_raw21_fivefold/fold{k}';model=joblib.load(run/'model.joblib');fit=json.loads((run/'training.json').read_text());assert sha(run/'model.joblib')==fit['model_sha256'] and fit['plan_sha256']==plan['validation_plan_sha256'] and np.array_equal(model.classes_,[0,1]) and list(model.feature_names_in_)==list(raw)
   saved=out/f'fold{k}.npz';assert sha(saved)==generated['folds'][k]['prediction_sha256'];q=np.load(saved);assert np.array_equal(q['id'],ids) and str(q['model_sha256'])==fit['model_sha256'] and str(q['final_plan_sha256'])==sha(planpath)
   p=np.concatenate([manual_probability(model,raw.iloc[j:j+16384]) for j in range(0,len(ids),16384)]);error=float(np.max(np.abs(p-q['pred'])));assert error<1e-12 and np.isfinite(p).all();rebuilt.append(p);proofs.append(dict(fold=k,full_test_manual_probability_error=error,model_sha256=fit['model_sha256']));del model
  avg=sum(rebuilt)/5;v12=np.clip(baseline.satisfaction.to_numpy(),1e-6,1-1e-6);a=np.clip(avg,1e-6,1-1e-6);score=.9375*(np.log(v12)-np.log1p(-v12))+.0625*(np.log(a)-np.log1p(-a));manual=1/(1+np.exp(-score));error=float(np.max(np.abs(manual-csv.satisfaction.to_numpy())));assert error<1e-14
  # Replay serialization using the original API probabilities, after their
  # independent full-row numerical reconstruction above, to prove exact bytes.
  from scipy.special import expit,logit
  api_mean=np.mean(np.stack([np.load(out/f'fold{k}.npz')['pred'] for k in range(5)]),axis=0);exact=expit(.9375*logit(v12)+.0625*logit(np.clip(api_mean,1e-6,1-1e-6)));buffer=io.StringIO();pd.DataFrame({'id':ids,'satisfaction':exact}).to_csv(buffer,index=False,float_format='%.17g');byte_sha=hashlib.sha256(buffer.getvalue().encode()).hexdigest();assert byte_sha==sha(csvpath)
  result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),final_plan_sha256=sha(planpath),audit_source_sha256=sha(__file__),rows=len(ids),CSV=csvpath.name,CSV_sha256=byte_sha,CSV_bytes=csvpath.stat().st_size,independent_manual_pool_error=error,CSV_bytes_replayed_exactly=True,fold_proofs=proofs,peak_RSS_bytes=max(peak[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),new_classifier_fits=0,parent_only_submitter=True,fresh_parent_quota_and_rank_required=True,actual_top500_verified=False)
  (ROOT/'research_r77_final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

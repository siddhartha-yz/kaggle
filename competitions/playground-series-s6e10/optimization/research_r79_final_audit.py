"""Independent native-adapter replay over every test row; byte proof before handoff."""
from research_r79_model import ROOT,configure,torch,np,TabSTARClassifier
from research_r79_audit import native_probability
from tabstar.training.lora import load_finetuned
from research_run_deadline import check_finalization
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from pathlib import Path
from datetime import datetime,timezone
from scipy.special import expit,logit
import json,hashlib,gc,threading,resource,pandas as pd,io,time
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 check_finalization();fp=ROOT/'research_r79_final_plan.json';plan=json.loads(fp.read_text())
 for n,h in plan['required_sha256'].items():assert sha(ROOT/n)==h,n
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/result.json').read_text())['eligible']
 audit=json.loads((ROOT/'research_r79_independent_audit.json').read_text());assert audit['status']=='passed' and audit['plan_sha256']==plan['validation_plan_sha256'];generated=json.loads((ROOT/'research_r79_final_inference.json').read_text());assert generated['final_plan_sha256']==sha(fp)
 configure();assert host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);out=ROOT/'research_artifacts/r79_final_inference';stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'audit_resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  raw=pd.read_csv(ROOT/'data/test.csv');ids=raw.pop('id').to_numpy();base=pd.read_csv(ROOT/plan['baseline_CSV'],float_precision='round_trip');sample=pd.read_csv(ROOT/'data/sample_submission.csv');csvpath=ROOT/plan['output_CSV'];csv=pd.read_csv(csvpath,float_precision='round_trip');assert raw.shape==(299844,21) and all(np.array_equal(ids,x.id) for x in [base,sample,csv]) and list(csv)==list(sample)==['id','satisfaction'] and sha(csvpath)==generated['CSV_sha256'];manual=[];api=[];proof=[]
  for k in range(5):
   check_finalization();run=ROOT/f'research_artifacts/r79_tabstar_raw21_fivefold/fold{k}';fit=json.loads((run/'training.json').read_text());assert sha(run/'model.joblib')==fit['model_sha256'];saved=out/f'fold{k}.npz';assert sha(saved)==generated['folds'][k]['prediction_sha256'];q=np.load(saved);assert np.array_equal(q['id'],ids) and str(q['final_plan_sha256'])==sha(fp);old=TabSTARClassifier.load(str(run/'model.joblib'));preprocessor=old.preprocessor_;del old;gc.collect();best=next((run/n).parent for n in fit['native_best_checkpoint_files'] if n.endswith('adapter_model.safetensors'))
   for n,h in fit['native_best_checkpoint_files'].items():assert sha(run/n)==h
   native=load_finetuned(str(best),'alana89/TabSTAR');native.to('cuda');parts=[];maximum=0.;softmax_error=0.
   for start in range(0,len(ids),16384):
    p,e=native_probability(native,preprocessor,raw.iloc[start:start+16384]);parts.append(p);maximum=max(maximum,float(np.max(np.abs(p-q['pred'][start:start+len(p)]))));softmax_error=max(softmax_error,e)
   assert maximum<1e-6;manual.append(np.concatenate(parts));api.append(q['pred']);proof.append(dict(fold=k,all_test_rows_replayed=len(ids),native_adapter_probability_error=maximum,two_logit_softmax_error=softmax_error));del native,preprocessor;gc.collect();torch.cuda.empty_cache();print('FINAL_AUDIT_FOLD',k,maximum,flush=True)
  a=np.clip(base.satisfaction.to_numpy(),1e-6,1-1e-6);b=np.clip(sum(manual)/5,1e-6,1-1e-6);reconstructed=1/(1+np.exp(-(.9375*(np.log(a)-np.log1p(-a))+.0625*(np.log(b)-np.log1p(-b)))));error=float(np.max(np.abs(reconstructed-csv.satisfaction.to_numpy())));assert error<1e-14
  original=expit(.9375*logit(a)+.0625*logit(np.clip(np.mean(np.stack(api),axis=0),1e-6,1-1e-6)));buffer=io.StringIO();pd.DataFrame({'id':ids,'satisfaction':original}).to_csv(buffer,index=False,float_format='%.17g');byte_sha=hashlib.sha256(buffer.getvalue().encode()).hexdigest();assert byte_sha==sha(csvpath)
  result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),final_plan_sha256=sha(fp),CSV=csvpath.name,CSV_sha256=byte_sha,CSV_bytes=csvpath.stat().st_size,rows=len(ids),full_manual_pool_error=error,CSV_bytes_replayed_exactly=True,fold_proofs=proof,seconds=time.monotonic()-tick,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_classifier_fits=0,parent_only_submitter=True,fresh_parent_quota_rank_required=True,actual_top500_verified=False);(ROOT/'research_r79_final_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

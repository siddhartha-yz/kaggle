"""One preregistered final CV ensemble; own trusted checkpoints only, no uploads."""
import os,time,gc,json,threading,subprocess
from pathlib import Path
from research_r4_lean_gpu_init import ROOT,now,dump,sha
import numpy as np,pandas as pd,torch,cloudpickle,joblib
from threadpoolctl import threadpool_limits

def main():
 plan=json.loads((ROOT/'research_r7_final_inference_plan.json').read_text());assert plan['runner_sha256']==sha(__file__)
 for name in ['r7_response_fivefold','r7_response_threefold']:
  result=json.loads((ROOT/'research_artifacts'/name/'result.json').read_text());assert result['status']=='complete' and result['eligible_for_new_partition_replication'],'Both frozen paired checks must qualify'
 out=ROOT/'research_artifacts/r7_response_cv_final';out.mkdir(exist_ok=True);torch.set_num_threads(6);torch.set_num_interop_threads(1);cap=778043392;assert torch.cuda.is_available();free,_=torch.cuda.mem_get_info();assert free>=cap+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 paths={'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
 for name,expected in paths.items():assert sha(ROOT/'data'/f'{name}.csv')==expected
 test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id);source=ROOT/'research_artifacts/r7_response_fivefold';record=json.loads((source/'plan.json').read_text());assert record['features_sha256']==sha(ROOT/'research_r7_response_features.py');peak=[0];violation=threading.Event();stop=threading.Event()
 def monitor():
  while not stop.wait(.5):
   try:
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=4)
    for line in q.stdout.splitlines():
     pid,mem=line.split(',')
     if int(pid)==os.getpid():
      peak[0]=max(peak[0],int(mem)*1024**2)
      if peak[0]>1073741824:violation.set()
   except (ValueError,subprocess.TimeoutExpired):pass
 thread=threading.Thread(target=monitor,daemon=True);thread.start();total=np.zeros(len(test),dtype='float64');receipts=[]
 try:
  for k in range(5):
   tr=joblib.load(source/f'fold{k}/transformer.joblib');assert list(raw)==tr.columns;V=tr.transform(raw)
   for j,seed in enumerate(record['seeds']):
    run=source/f'fold{k}/member{j}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['config']==dict(record['config'],random_state=seed);assert rec['model_sha256']==sha(run/'model.pkl');cache=out/f'fold{k}_member{j}.npz';assert not violation.is_set();free,_=torch.cuda.mem_get_info();assert free>=1024**3;tick=time.monotonic()
    if cache.exists():
     z=np.load(cache);assert np.array_equal(z['id'],test.id);pred=z['pred']
    else:
     with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
     assert np.array_equal(model.classes_,[0,1]);model.to('cuda:0')
     with threadpool_limits(6):pred=model.predict_proba(V)[:,1]
     assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all();np.savez_compressed(cache,id=test.id.to_numpy(),pred=pred);del model;gc.collect();torch.cuda.empty_cache()
    total+=pred;receipts.append({'fold':k,'member':j,'model_sha256':rec['model_sha256'],'prediction_sha256':sha(cache),'elapsed_seconds':time.monotonic()-tick});dump(out/'state.json',{'status':'predicting','updated_utc':now(),'completed_models':len(receipts),'models':receipts,'gpu_process_peak_bytes':peak[0]});print('PREDICTED',k,j,flush=True)
   del tr,V;gc.collect()
  assert len(receipts)==40;nn=total/40;old=np.load(ROOT/'research_artifacts/gpu_value_v6_full/test_components.npz');assert np.array_equal(old['id'],test.id);assert sha(ROOT/'submission_realmlp_value_v6.csv')=='36ea4ef80e1dd1911257f240391d9cdb8474b8b5deb5c52733582da80bdd1aaa';old_error=float(np.max(np.abs(.75*old['v5']+.25*old['realmlp']-old['candidate'])));assert old_error<1e-12;pred=.5*old['v5']+.5*nn;assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all() and np.std(pred)>.01;np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v5=old['v5'],realmlp=nn,candidate=pred);csv=ROOT/'submission_response_lean_r7_v9.csv'
  if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':pred}).to_csv(csv,index=False,float_format='%.17g')
  check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==len(test) and np.array_equal(check.id,test.id);err=float(np.max(np.abs(check.satisfaction.to_numpy()-pred)));assert err<1e-14;assert not violation.is_set();result={'status':'complete','completed_utc':now(),'csv':csv.name,'rows':len(check),'columns':list(check),'sha256':sha(csv),'bytes':csv.stat().st_size,'positive_class':True,'members':40,'formula':'.5*v5_full+.5*mean(5folds x8responseNN)','no_test_labels_used':True,'no_test_fitted_statistics':True,'old_v6_reconstruction_error':old_error,'csv_roundtrip_max_error':err,'gpu_process_peak_bytes':peak[0],'formal_current_round_used':1,'formal_remaining':1,'formal_total_limit':2,'parent_only_submitter':True,'not_actual_leaderboard_confirmation':True,'plan_sha256':sha(ROOT/'research_r7_final_inference_plan.json')};dump(out/'result.json',result);dump(ROOT/'research_r7_state.json',dict(result,active_training=False,route='candidate_ready_for_parent_upload'));print(json.dumps(result,indent=2),flush=True)
 finally:stop.set();thread.join(timeout=5)
if __name__=='__main__':main()

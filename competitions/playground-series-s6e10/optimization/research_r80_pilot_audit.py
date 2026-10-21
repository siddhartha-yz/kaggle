"""Inner-only model/selection replay. Does not qualify an outer candidate."""
from research_r79r2_model import ROOT,configure,torch,np
from research_r79r2_audit import inner_split,parameter_hash,native_probability,verify_preprocessor
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from tabstar.tabstar_model import TabSTARClassifier
from tabstar.tabstar_verbalizer import TabSTARVerbalizer
from tabstar.training.lora import load_finetuned
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,threading,resource,gc,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((ROOT/'research_r80_plan.json').read_text());ap=json.loads((ROOT/'research_r80_pilot_audit_plan.json').read_text());ph=sha(ROOT/'research_r80_plan.json');assert ap['main_plan_sha256']==ph and ap['audit_source_sha256']==sha(__file__)
 for n,h in {**p['source_sha256'],**p['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
 for n,h in p['native_package_sources_sha256'].items():assert sha(ROOT.parent/n)==h,n
 out=ROOT/'research_artifacts/r80_tabstar_3v12_inner_control';result=json.loads((out/'result.json').read_text());fit=json.loads((out/'training.json').read_text());assert result['status']=='complete_inner_control' and result['plan_sha256']==fit['plan_sha256']==ph and result['epochs_completed']==12 and not result['outer_quality_scored'];assert all(sha(out/n)==h for n,h in result['files_sha256'].items());assert fit['parameters']==p['parameters_12']
 trace=fit['trace'];assert len(trace['epochs'])==len(trace['inner_evaluations'])==12
 for k,e in enumerate(trace['epochs']):assert e['rows']==558706 and e['microbatches']==8730 and e['updates_called']==4365*(k+1) and 0<e['updates_executed']<=e['updates_called'] and abs(e['optimizer_groups'][0]['lr']-p['epoch_start_LRs']['12'][k])<1e-15
 best=int(np.argmax([x['metric'] for x in trace['inner_evaluations']]));assert fit['best_epoch']==result['new12_native_best_epoch']==best+1
 frame=pd.read_csv(ROOT/'data/train.csv');raw=frame.drop(columns=['id','satisfaction']);ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');fi,vi=next(StratifiedKFold(5,shuffle=True,random_state=20261018).split(raw,y));ti,iv=inner_split(fi,y);ix=np.load(out/'indices.npz')
 for n,a in [('fi',fi),('vi',vi),('inner_train',ti),('inner_valid',iv),('inner_train_id',ids[ti]),('inner_valid_id',ids[iv])]:assert np.array_equal(ix[n],a)
 assert len(iv)==1002 and not np.isin(np.r_[ti,iv],vi).any();z=np.load(out/'new12_inner.npz');old=np.load(out/'existing3_inner.npz')
 for a in [z,old]:assert all(np.array_equal(a[n],v) for n,v in [('idx',iv),('id',ids[iv]),('target',y[iv])]) and np.isfinite(a['pred']).all()
 configure();assert host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'pilot_audit_resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  saved=TabSTARClassifier.load(str(out/'model.joblib'));assert saved.max_epochs==saved.patience==12 and saved.cp_average is False and list(saved.classes_)==['False','True'];labels=pd.Series(np.where(y,'True','False'),name='satisfaction');rebuilt=TabSTARVerbalizer(is_cls=True);rebuilt.fit(raw.iloc[ti],labels.iloc[ti]);verify_preprocessor(rebuilt,saved.preprocessor_,raw,ti)
  for n,h in fit['native_best_checkpoint_files'].items():assert sha(out/n)==h
  bestdir=next((out/n).parent for n in fit['native_best_checkpoint_files'] if n.endswith('adapter_model.safetensors'));native=load_finetuned(str(bestdir),'alana89/TabSTAR');assert parameter_hash(native)==parameter_hash(saved.model_)==fit['final_parameters_sha256'] and parameter_hash(native,True)==p['pretrained_base_parameters_sha256'];native.to('cuda');pred,softmax_error=native_probability(native,rebuilt,raw.iloc[iv]);error=float(np.max(np.abs(pred-z['pred'])));assert error<1e-6 and abs(roc_auc_score(y[iv],pred)-trace['inner_evaluations'][best]['metric'])<1e-4
  oldfit=json.loads((ROOT/'research_artifacts/r79r2_tabstar_raw21_fivefold/fold0/training.json').read_text());oldbest=int(np.argmax([a['metric'] for a in oldfit['trace']['inner_evaluations']]));a=oldfit['trace']['inner_evaluations'][oldbest];b=trace['inner_evaluations'][best];worth=bool(best+1>3 and b['metric']>a['metric'] and b['loss']<a['loss']);assert worth==result['worth_prospective_outer_validation'];report=dict(status='passed_inner_control_verification',checked_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=ph,audit_plan_sha256=sha(ROOT/'research_r80_pilot_audit_plan.json'),strict_own_inner_indices_exact=True,own_scalers_quantiles_rebuilt_exact=True,native_best_adapter_parameters_exact=True,pretrained_backbone_unchanged=True,inner_prediction_replay_error=error,independent_softmax_error=softmax_error,selected_epoch=best+1,worth_prospective_outer_validation=worth,outer_candidate_qualified=False,outer_quality_scored=False,seconds=time.monotonic()-tick,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,statistics_limitation=p['statistics_limitation']);(ROOT/'research_r80_pilot_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

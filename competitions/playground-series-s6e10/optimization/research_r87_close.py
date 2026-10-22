"""Close failed R87 without replaying training or changing the frozen recipe."""
import json,shutil,numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import t
from research_r87_common import ROOT,OUT,sha,dump,now,verify,refs
p=verify();r=json.loads((ROOT/'research_r87_result.json').read_text());assert r['status']=='closed_negative';models=[];checks={}
for stage,s in r['stages'].items():
 z=np.load(OUT/stage/'oof.npz');b=refs(stage);assert sha(OUT/stage/'oof.npz')==s['oof_sha256']
 for key in ['id','target','fold']:assert np.array_equal(z[key],b[0][key])
 assert np.array_equal(z['reference_v12'],b[2]['candidate']);d=[];K=len(np.unique(z['fold']))
 for k in range(K):
  run=OUT/stage/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r87_plan.json') and rec['config']==p['config'] and rec['feature_width']==75 and rec['original74_query_inputs_exact'] and not rec['validation_used_for_fit_or_epoch_selection']
  for n,h in rec['files_sha256'].items():assert sha(run/n)==h
  assert len(rec['epochs'])==3 and rec['model_reload_error']<1e-7
  assert rec['peak_GPU_bytes']<=p['resource_limits']['own_GPU_ceiling_bytes'] and rec['peak_RSS_bytes']<=p['resource_limits']['own_RSS_ceiling_bytes']
  assert all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=p['resource_limits']['allocator_cap_bytes'] for e in rec['epochs'])
  v=np.load(run/'validation.npz');ix=np.flatnonzero(z['fold']==k);fi=np.flatnonzero(z['fold']!=k);split=np.load(run/'indices.npz')
  assert np.array_equal(split['fi'],fi) and np.array_equal(split['vi'],ix) and np.array_equal(v['id'],z['id'][ix]) and np.array_equal(v['idx'],ix) and np.array_equal(v['target'],z['target'][ix]) and np.array_equal(v['pred'],z['pred'][ix])
  d.append(roc_auc_score(z['target'][ix],z['candidate'][ix])-roc_auc_score(z['target'][ix],z['reference_v12'][ix]));models.append(dict(stage=stage,fold=k,seconds=rec['elapsed_seconds'],peak_GPU_bytes=rec['peak_GPU_bytes'],peak_RSS_bytes=rec['peak_RSS_bytes'],model_sha256=rec['files_sha256']['model.pkl']))
 auc=roc_auc_score(z['target'],z['candidate']);ev=s['paired_evidence'];assert abs(auc-ev['pooled']['candidate_auc'])<1e-12;lower=float(np.mean(d)-t.ppf(.975,K-1)*np.sqrt((1/K+1/(K-1))*np.var(d,ddof=1)));assert abs(lower-ev['corrected_resampled_t']['lower_bound'])<1e-12
 checks[stage]=dict(candidate_AUC=auc,delta_AUC=ev['pooled']['delta_auc'],positive_folds=sum(x>0 for x in d),correctedt_lower=lower,oof_sha256=sha(OUT/stage/'oof.npz'))
assert not (OUT/'threefold').exists()
conclusion=dict(status='completed_closed_recipe',completed_utc=now(),plan_sha256=sha(ROOT/'research_r87_plan.json'),result_sha256=sha(ROOT/'research_r87_result.json'),stage_integrity=checks,validation=r['stages'],models=models,resource_interruption=False,interpretation='Allfivefoldnegative;fixedofficialteacherprobabilityadditiontocurrentNN74rejected. No prob/logit/weight/epochrescue. Doesnotestablishallpossibleteacherintegrationsuseless.',new_formal_submissions=0,new_test_inference=False,full_refit=False,own_output_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),disk_available_bytes=shutil.disk_usage(ROOT).free,historical_selection_limitation=p['statistics_limitation'],campaign_still_authorized=True,actual_top500_verified=False)
dump(ROOT/'research_r87_conclusion.json',conclusion);print(json.dumps({k:conclusion[k] for k in ['status','stage_integrity','own_output_bytes','disk_available_bytes']}))

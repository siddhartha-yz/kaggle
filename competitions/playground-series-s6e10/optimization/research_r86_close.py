"""Close a failed frozen recipe with independent identity and numerical checks."""
import json,numpy as np,shutil,joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import t
from research_r86_common import ROOT,OUT,sha,dump,now,verify,refs
plan=verify();r=json.loads((ROOT/'research_r86_result.json').read_text());assert r['status']=='closed_negative'
checks={};models=[]
for stage,s in r['stages'].items():
 z=np.load(OUT/stage/'oof.npz');ref=refs(stage);assert sha(OUT/stage/'oof.npz')==s['oof_sha256']
 for key in ('id','target','fold'):assert np.array_equal(z[key],ref[0][key])
 assert np.array_equal(z['reference_v12'],ref[2]['candidate']);e=s['paired_evidence'];auc=roc_auc_score(z['target'],z['candidate']);assert abs(auc-e['pooled']['candidate_auc'])<1e-12
 d=[];K=len(np.unique(z['fold']))
 for k in range(K):
  p=OUT/stage/f'fold{k}';rec=json.loads((p/'result.json').read_text());assert rec['status']=='complete' and rec['unchanged_first84_exact'] and rec['features']==84 and rec['plan_sha256']==sha(ROOT/'research_r86_plan.json')
  for n,h in rec['artifact_sha256'].items():assert sha(p/n)==h
  v=np.load(p/'validation.npz');ix=np.flatnonzero(z['fold']==k);fi=np.flatnonzero(z['fold']!=k);assert np.array_equal(v['idx'],ix) and np.array_equal(v['id'],z['id'][ix]) and np.array_equal(v['target'],z['target'][ix]) and np.array_equal(v['pred'],z['pred'][ix])
  tr=joblib.load(p/'transformer.joblib');assert tr.base.n==len(fi) and tr.base.te_columns==[] and tr.columns==plan['feature_columns']
  assert rec['parameters']==plan['learner_parameters'] and rec['perfeature_bins_native_verified']
  d.append(roc_auc_score(z['target'][ix],z['candidate'][ix])-roc_auc_score(z['target'][ix],z['reference_v12'][ix]));models.append(dict(stage=stage,fold=k,seconds=rec['seconds'],peak_rss_bytes=rec['peak_rss_bytes'],model_sha256=rec['artifact_sha256']['model.txt']))
 lower=float(np.mean(d)-t.ppf(.975,K-1)*np.sqrt((1/K+1/(K-1))*np.var(d,ddof=1)));assert abs(lower-e['corrected_resampled_t']['lower_bound'])<1e-12
 checks[stage]=dict(AUC=auc,delta_AUC=e['pooled']['delta_auc'],positive_folds=sum(x>0 for x in d),folds=K,correctedt_lower=lower,oof_sha256=sha(OUT/stage/'oof.npz'))
assert not (OUT/'full').exists() and not (ROOT/'submission_teacher_bins1023_r86_v13.csv').exists()
conclusion=dict(status='completed_closed_recipe',experiment='R86_only_teacher_histogram_resolution255to1023_LGB84',completed_utc=now(),first_real_start_utc='2026-10-21T12:21:58.528865+00:00',controller_pid=674395,first_worker_pid=674406,service='codex-kaggle-r86-validation.service',plan_sha256=sha(ROOT/'research_r86_plan.json'),result_sha256=sha(ROOT/'research_r86_result.json'),stage_integrity=checks,validation={stage:s['paired_evidence'] for stage,s in r['stages'].items()},models=models,independent_integrity_check='Allidentity/sourcehashes/modelartifacts/OOFcoverage/sklearnAUC/correctedtformula/ownouterencodermean verified',interpretation='Frozen minimum-change recipe didnotmeet gate; close without rescue tuning. Doesnotprove allhistogramresolutionchangesuseless.',full_refit=False,CSV_generated=False,new_formal_submissions=0,weight_search=False,GPU_used=False,own_output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),disk_available_bytes=shutil.disk_usage(ROOT).free,resource_interruption=False,limitation='Historicallyreusedlabels,includingchosenafterR83/R84; no untouchedholdout or selectioncorrection',campaign_still_authorized=True,goal='Actualpublicrank<=500 remainsunachieved; thisisonecompletedrecipe,notprojectcompletion')
dump(ROOT/'research_r86_conclusion.json',conclusion);print(json.dumps({k:conclusion[k] for k in ['status','plan_sha256','result_sha256','stage_integrity','own_output_bytes','disk_available_bytes']}))

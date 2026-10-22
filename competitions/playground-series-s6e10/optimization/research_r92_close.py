"""Integrity closeout of failed R92; no model inference, new fitting or method rescue."""
import json,hashlib,shutil,numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from scipy.stats import t
from research_r92_common import ROOT,OUT,plan,refs,sha,dump,now,write_guard,candidate
p=plan(all_inputs=True);r=json.loads((ROOT/'research_r92_result.json').read_text());assert r['status']=='closed_not_qualified' and set(r['stages'])=={'fivefold'}
s=r['stages']['fivefold'];ev=s['vs_actual_v12'];assert not s['eligible'] and not ev['eligible_for_new_partition_replication'];tree,ce,base=refs('fivefold');z=np.load(OUT/'fivefold/oof.npz');assert sha(OUT/'fivefold/oof.npz')==s['oof_sha256']
for key in ['id','target','fold']:assert np.array_equal(z[key],base[key])
assert np.array_equal(z['reference_v12'],base['candidate']);assert np.max(np.abs(candidate(ce['candidate'],z['pred'])-z['candidate']))<1e-12
records=[];d=[];total_seconds=0
for k in range(5):
 run=OUT/'fivefold'/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r92_plan.json') and rec['new_neural_fits']==0 and rec['native_members']==32 and rec['origins_count']==32 and rec['slope_fits']==rec['weight_fits']==0 and not rec['official_validation_labels_used_for_calibration']
 for name,digest in rec['files_sha256'].items():assert sha(run/name)==digest
 fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);orig=json.loads((run/'origins.json').read_text());b=np.asarray(orig['bias'],'float64')
 assert b.shape==(32,) and np.isfinite(b).all() and orig['training_rows']==len(fi) and orig['training_prior']==float(base['target'][fi].mean()) and orig['training_indices_sha256']==hashlib.sha256(fi.tobytes()).hexdigest() and not orig['validation_labels_used'] and orig['moment_error']<=1e-10
 assert orig['source_model_sha256']==p['models']['fivefold'][k]['files_sha256']['model.pkl']
 val=np.load(run/'validation.npz');assert np.array_equal(val['idx'],vi) and np.array_equal(val['id'],base['id'][vi]) and np.array_equal(val['pred'],z['pred'][vi]) and np.array_equal(val['legacy_replay'],z['legacy_replay'][vi])
 f=np.load(run/'training_fixture.npz');assert np.array_equal(f['idx'],fi[:256]) and np.array_equal(f['id'],base['id'][fi[:256]]) and f['margin'].shape==(32,256)
 trainerr=float(np.max(np.abs(expit(f['margin']+b[:,None]).mean(0)-f['new_probability'])));assert trainerr<1e-12
 g=np.load(run/'validation_fixture.npz');assert np.array_equal(g['idx'],vi[:256]) and g['margin'].shape==(32,256);valerr=float(np.max(np.abs(expit(g['margin']+b[:,None]).mean(0)-val['pred'][:256])));assert valerr<1e-12
 shift=np.linspace(-4.,4.,32);inv=float(np.max(np.abs(expit(g['margin']+shift[:,None]+(b-shift)[:,None]).mean(0)-g['new_probability'])));assert inv<1e-12
 assert rec['own_training_raw_replay_error']<=2e-6 and rec['outer_legacy_replay_error']<=2e-6 and rec['training_moment_error']<=1e-10 and rec['origin_invariance_error']<1e-12
 assert rec['peak_GPU_bytes']<=p['resources']['GPU_process_ceiling_bytes'] and rec['peak_RSS_bytes']<=p['resources']['RSS_ceiling_bytes']
 d.append(float(roc_auc_score(base['target'][vi],z['candidate'][vi])-roc_auc_score(base['target'][vi],base['candidate'][vi])));total_seconds+=rec['seconds']
 records.append(dict(fold=k,train_rows=len(fi),valid_rows=len(vi),seconds=rec['seconds'],fixture_train_formula_error=trainerr,fixture_validation_formula_error=valerr,fixture_origin_invariance_error=inv,moment_error_reported=rec['training_moment_error'],bias_range=rec['bias_range'],own_training_operator_change_max=rec['own_training_operator_change_max'],peak_GPU_bytes=rec['peak_GPU_bytes'],peak_RSS_bytes=rec['peak_RSS_bytes'],source_model_sha256=orig['source_model_sha256']))
auc=float(roc_auc_score(z['target'],z['candidate']));assert abs(auc-ev['pooled']['candidate_auc'])<1e-12
lower=float(np.mean(d)-t.ppf(.975,4)*np.sqrt((1/5+1/4)*np.var(d,ddof=1)));assert abs(lower-ev['corrected_resampled_t']['lower_bound'])<1e-12
noop_delta=float(roc_auc_score(base['target'],candidate(ce['candidate'],z['legacy_replay']))-roc_auc_score(base['target'],base['candidate']));assert abs(noop_delta-s['no_op_control_delta_AUC'])<1e-12 and abs(noop_delta)<=1e-7
assert not (OUT/'threefold').exists() and not (OUT/'test').exists();best=ROOT/'submission_pure_ranknet_r68_v12.csv';assert sha(best)=='f48276b144e8c9297fb9fec27174044a55a7eb3f2af23dde3f13478a38a4e545'
probe=json.loads((OUT/'probe/result.json').read_text());assert probe['status']=='passed_no_quality_score' and probe['current_raw_probability_replay_error']==0 and probe['normal_restore_error']==probe['exception_restore_error']==0
artifact_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())+sum(f.stat().st_size for f in ROOT.glob('research_r92*') if f.is_file());assert artifact_bytes<128*2**20
record=dict(status='completed_closed_not_qualified',completed_utc=now(),plan_sha256=sha(ROOT/'research_r92_plan.json'),result_sha256=sha(ROOT/'research_r92_result.json'),oof_sha256=s['oof_sha256'],integrity_check_kind='Closedrecipefile/row/formula/fixture/statisticsverification;notpromotionaudit',all_source_model_and_input_hashes_verified=True,all_fivefold_rows_covered_once=True,positive_folds=sum(x>0 for x in d),fold_deltas=d,validation=ev,no_op_control_delta_AUC=noop_delta,probe=probe,fold_records=records,stage_worker_seconds_sum=total_seconds,new_neural_fits=0,new_member_intercepts_fitted=160,slope_fits=0,weight_fits=0,threefold_run=False,promotion_audit_run=False,new_test_inference=False,new_CSV_generated=False,new_formal_submissions=0,resource_interruption=False,artifact_bytes_before_closeout=artifact_bytes,artifact_cap_bytes=128*2**20,disk_available_bytes=shutil.disk_usage(ROOT).free,best_v12_CSV_unchanged=True,interpretation='Memberorigininvariance/replayisimplementable, butfrozenAUCgainisnotqualified. Pooled+3.533ppm vsmeanwithinfolddelta+.649ppm;crossfoldscorealignmentcancontributetopooledAUC. Do notrescuewithslopes/weights/alternativeaggregation.',limitations=p['limitations'],campaign_still_authorized=True,actual_top500_verified=False)
write_guard(1);dump(ROOT/'research_r92_conclusion.json',record);print(json.dumps({k:record[k] for k in ['status','positive_folds','no_op_control_delta_AUC','oof_sha256','artifact_bytes_before_closeout','disk_available_bytes']}))

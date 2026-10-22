"""Close failed frozen gates after full paired OOF; integrity only, no new fits or searches."""
from research_r94_common import *
import numpy as np

def main():
 p=plan(inputs=True);ph=sha(ROOT/'research_r94_full_plan.json');out=OUT/'fivefold'
 result=json.loads((out/'result.json').read_text());driver=json.loads((ROOT/'research_r94_driver_state.json').read_text())
 assert result['status']=='complete_frozen_paired_quality' and result['plan_sha256']==ph and not result['eligible']
 assert driver['status']=='closed_not_qualified' and driver['result_sha256']==sha(out/'result.json')
 assert not (OUT/'threefold').exists()
 base=np.load(ROOT/'research_artifacts/r68_pure_ranknet_fivefold/oof.npz');z=np.load(out/'oof.npz')
 assert sha(out/'oof.npz')==result['oof_sha256']
 assert all(np.array_equal(z[k],base[k]) for k in ['id','target','fold'])
 assert len(z['id'])==699635
 records=[];checkpoints=[]
 for k in range(5):
  run=out/f'fold{k}';d=json.loads((run/'complete.json').read_text());assert d['plan_sha256']==ph
  assert not d['outer_labels_parsed'] and not d['outer_quality_scored']
  for n,h in d['files_sha256'].items():assert sha(run/n)==h,n
  pair=[]
  for arm in ['scratch','rtd']:
   f=json.loads((run/f'{arm}_complete.json').read_text());assert f['initial_hash']==d['initial_hash']
   epochs=[e for e in f['records'] if e['phase']=='supervised'];pre=[e for e in f['records'] if e['phase']=='pretrain']
   assert len(epochs)==20 and len(pre)==(5 if arm=='rtd' else 0)
   assert f['best_epoch']==min(epochs,key=lambda e:e['inner_logloss'])['epoch']
   assert np.allclose([e['lr'] for e in epochs],p['supervised_schedule_epoch_start_LRs'],rtol=0,atol=1e-15)
   pair.append([e['epoch_seed'] for e in epochs])
   checkpoints.append(dict(fold=k,arm=arm,best_epoch=f['best_epoch'],inner_logloss=f['best_inner_logloss']))
  assert pair[0]==pair[1];records.append(d)
 csv=ROOT/'submission_pure_ranknet_r68_v12.csv'
 assert sha(csv)=='f48276b144e8c9297fb9fec27174044a55a7eb3f2af23dde3f13478a38a4e545'
 comparisons={name:dict(delta_AUC=v['pooled']['delta_auc'],paired_95pct_CI=v['pooled']['conditional_95pct_ci'],
   corrected_resampled_t_lower=v['corrected_resampled_t']['lower_bound'],
   fold_deltas=[f['delta_auc'] for f in v['fold_comparisons']],dual_gates_pass=v['eligible_for_new_partition_replication'])
   for name,v in result['comparisons'].items()}
 conclusion=dict(status='closed_fivefold_not_qualified_no_threefold_or_submission',closed_utc=now(),
   full_plan_sha256=ph,result_sha256=sha(out/'result.json'),OOF_sha256=sha(out/'oof.npz'),
   driver_state_sha256=sha(ROOT/'research_r94_driver_state.json'),
   completed_pairs=5,OOF_rows=len(z['id']),AUC=result['AUC'],comparisons=comparisons,
   fixed_logit_mix=result['fixed_logit_mix'],all_pairs_before_scoring=True,
   checkpoint_selection=checkpoints,
   interpretation=dict(
     mechanism='Raw RTD improves pooled AUC by0.00028175 over the paired same-architecture supervised control, with all5fold deltas positive. Paired pooled CI excludes0 but corrected-fold lower is negative; do not claim robust replication.',
     complementarity='Fixed RTD/v12 blend gains only0.000003811 over v12,2/5folds positive; both admission bounds fail. Better standalone quality is not sufficient incremental ensemble value.',
     paired_blend='RTD blend exceeds scratch blend by0.000017012, but corrected-fold bound fails. Neither required comparison passes both frozen gates.',
     decision='Close this exact recipe; no threefold, promotion audit, test prediction, full-data fit, new submission, weight rescue, or derived seed/epoch/architecture variants.'),
   integrity=dict(status='passed_closeout_integrity_not_promotion_audit',all_source_and_input_hashes_verified=True,
     all_five_pair_artifact_hashes_verified=True,paired_initial_states_and_supervised_epoch_seeds_equal=True,
     inner_only_checkpoint_selection_verified=True,outer_labels_excluded_by_worker_parser=True,
     original_v12_CSV_sha256=sha(csv),new_fits_for_closeout=0),
   resources=dict(controller_elapsed_seconds=driver['elapsed_seconds'],
     maximum_worker_GPU_process_bytes=max(d['peak_process_GPU_bytes'] for d in records),
     maximum_worker_RSS_bytes=max(d['peak_RSS_bytes'] for d in records),
     artifact_bytes_before_closeout=ownbytes(),artifact_cap_bytes=192*2**20,
     disk_free_bytes=shutil.disk_usage(ROOT).free,disk_reserve_bytes=2*2**30,
     own_resource_failure_files=[str(f.relative_to(ROOT)) for f in OUT.rglob('resource_failure.json')]),
   limitations=['Historically reused official labels; conditional intervals omit adaptive search history.','No untouched-holdout or actual ranking improvement claim.','Result concerns one fixed small RTD implementation and budget, not all RTD mechanisms.'],
   new_submissions=0,new_test_predictions=False,main_unchanged=True,project_top500_goal_completed=False)
 assert not conclusion['resources']['own_resource_failure_files']
 reserve();dump(ROOT/'research_r94_conclusion.json',conclusion)
 dump(ROOT/'research_r94_fivefold_result.json',result)
 print(json.dumps(dict(status=conclusion['status'],conclusion_sha256=sha(ROOT/'research_r94_conclusion.json'),artifact_MiB=ownbytes()/2**20)))

if __name__=='__main__':main()

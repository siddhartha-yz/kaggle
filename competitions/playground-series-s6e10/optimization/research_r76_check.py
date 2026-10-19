"""Verify all folds and native averaging before evaluating one fixed v12 pool."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,sys,numpy as np
from scipy.special import expit,logit
def main(stage):
 plan=json.loads((ROOT/'research_r76_plan.json').read_text());ph=sha(ROOT/'research_r76_plan.json');out=ROOT/f'research_artifacts/r76_mitra_v2_{stage}';base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');pred=np.full(len(base['id']),np.nan)
 for k in sorted(np.unique(base['fold'])):
  run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==ph and rec['runner_sha256']==plan['source_sha256']['research_r76_worker.py'] and all(sha(run/n)==h for n,h in rec['files_sha256'].items());fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);indices=np.load(run/'outer_indices.npz');assert np.array_equal(indices['fi'],fi) and np.array_equal(indices['vi'],vi) and np.array_equal(indices['train_id'],base['id'][fi]) and np.array_equal(indices['valid_id'],base['id'][vi]);natives=[]
  for seed in plan['training']['seeds']:
   own=run/f'native{seed}';fit=json.loads((own/'training.json').read_text());done=json.loads((own/'result.json').read_text())
   if fit['plan_sha256']!=ph:
    legacy=plan['engineering_recovery']['accepted_training'][f'{stage}/fold{k}/native{seed}'];assert fit['plan_sha256']==legacy['plan_sha256'] and sha(own/'training.json')==legacy['training_metadata_sha256'] and fit['files_sha256']==legacy['files_sha256']
   if done['plan_sha256']!=ph:assert sha(own/'result.json')==plan['engineering_recovery']['accepted_native_results'][f'{stage}/fold{k}/native{seed}']['result_sha256']
   assert fit['completed_steps']==50 and not fit['outer_validation_used'] and done['parameters_unchanged_during_inference'] and done['training_metadata_sha256']==sha(own/'training.json')
   assert max(done['repeat_error'],done['cache_first1024_error'],done['cache_last_query_error'])<1e-6 and all(sha(own/n)==h for n,h in {**fit['files_sha256'],**done['files_sha256']}.items());ep=np.load(own/'episodes.npz');assert ep['own_positions'].shape==(50,1024) and np.array_equal(ep['global_idx'],fi[ep['own_positions']]) and np.array_equal(ep['global_id'],base['id'][ep['global_idx']]) and all(len(np.unique(a))==1024 for a in ep['own_positions']);ctx=np.load(own/'context.npz');assert len(ctx['global_idx'])==4096 and len(np.unique(ctx['global_idx']))==4096 and np.isin(ctx['global_idx'],fi).all() and np.array_equal(ctx['target'],base['target'][ctx['global_idx']]) and np.bincount(ctx['target']).tolist()==[2048,2048]
   z=np.load(own/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],base['id'][vi])
   for start in range(0,len(vi),1024):
    q=np.load(own/f'query{start:06d}.npz');take=vi[start:start+1024];assert str(q['plan_sha256'])==done['plan_sha256'] and np.array_equal(q['idx'],take) and np.array_equal(q['id'],base['id'][take]) and np.array_equal(q['pred'],z['pred'][start:start+len(take)])
   natives.append(z['pred'])
  final=np.load(run/'validation.npz');assert np.array_equal(final['idx'],vi) and np.array_equal(final['id'],base['id'][vi]) and np.array_equal(final['target'],base['target'][vi]) and np.array_equal(final['pred'],np.mean(natives,axis=0));pred[vi]=final['pred']
 assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,candidate=candidate,reference_v12=base['candidate']);proof=evidence(base['target'],base['candidate'],candidate,base['fold']);record=dict(status='complete',stage=stage,checked_utc=now(),eligible=proof['eligible_for_new_partition_replication'],vs_formal_v12=proof,all_folds_before_scoring=True,plan_sha256=ph,oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',record);print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

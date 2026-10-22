"""Score only complete paired OOFs; mechanism and existing-blend utility stay distinct."""
from research_r94_common import *
from research_r4_common import evidence
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
import numpy as np,sys

def mix(base,new):return expit(.875*logit(np.clip(base,1e-6,1-1e-6))+.125*logit(np.clip(new,1e-6,1-1e-6)))
def main(stage):
 p=plan(inputs=True);ph=sha(ROOT/'research_r94_full_plan.json');K=5 if stage=='fivefold' else 3
 out=OUT/stage
 # No official quality labels accessed until all completed-fit hashes are verified.
 records=[]
 for k in range(K):
  run=out/f'fold{k}';d=json.loads((run/'complete.json').read_text())
  assert d['status']=='complete_paired_outerfit' and d['stage']==stage and d['fold']==k and d['plan_sha256']==ph
  assert not d['outer_labels_parsed'] and not d['outer_quality_scored']
  for n,h in d['files_sha256'].items():assert sha(run/n)==h,n
  for arm in ['scratch','rtd']:
   a=json.loads((run/f'{arm}_complete.json').read_text());assert a['initial_hash']==d['initial_hash'] and a['plan_sha256']==ph
   assert a['best_sha256']==sha(run/f'{arm}_best.pt') and len([v for v in a['records'] if v['phase']=='supervised'])==20
   pre=[v for v in a['records'] if v['phase']=='pretrain'];assert len(pre)==(5 if arm=='rtd' else 0)
   supervised=[v for v in a['records'] if v['phase']=='supervised']
   assert a['best_epoch']==min(supervised,key=lambda x:x['inner_logloss'])['epoch']
   assert a['best_inner_logloss']==min(v['inner_logloss'] for v in supervised)
   assert not a['outer_labels_parsed']
  records.append(d)
 base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz')
 ids,y,fold,reference=[base[n] for n in ['id','target','fold','candidate']]
 scratch=np.full(len(ids),np.nan,dtype='float32');rtd=scratch.copy()
 for k in range(K):
  z=np.load(out/f'fold{k}/predictions.npz');ix=np.flatnonzero(fold==k)
  assert np.array_equal(z['idx'],ix) and np.array_equal(z['id'],ids[ix])
  scratch[ix]=z['scratch'];rtd[ix]=z['rtd']
 assert np.isfinite(scratch).all() and np.isfinite(rtd).all()
 cs,cr=mix(reference,scratch),mix(reference,rtd)
 comparisons=dict(
  raw_RTD_vs_same_architecture_scratch=evidence(y,scratch,rtd,fold),
  RTD_blend_vs_v12=evidence(y,reference,cr,fold),
  RTD_blend_vs_scratch_blend=evidence(y,cs,cr,fold),
  scratch_blend_vs_v12_descriptive=evidence(y,reference,cs,fold))
 eligible=all(comparisons[n]['eligible_for_new_partition_replication'] for n in ['RTD_blend_vs_v12','RTD_blend_vs_scratch_blend'])
 reserve(32*2**20);target=out/'oof.npz';tmp=out/'.oof.npz'
 np.savez_compressed(tmp,id=ids,target=y,fold=fold,scratch=scratch,rtd=rtd);tmp.replace(target)
 result=dict(status='complete_frozen_paired_quality',stage=stage,checked_utc=now(),plan_sha256=ph,all_pairs_before_quality=True,
   fixed_logit_mix=dict(v12=.875,new_model=.125,clip=1e-6),
   AUC={n:float(roc_auc_score(y,v)) for n,v in [('v12',reference),('scratch_raw',scratch),('RTD_raw',rtd),('scratch_blend',cs),('RTD_blend',cr)]},
   comparisons=comparisons,eligible=eligible,
   admission_rule='Both original dual gates on RTD blend vs v12 AND RTD blend vs scratch blend; raw standalone mechanism result reported separately, not an additional selected gate.',
   oof_sha256=sha(target),completed_pair_sha256=[sha(out/f'fold{k}/complete.json') for k in range(K)],artifact_bytes=ownbytes(),
   limitations='Historically reused labels. These intervals omit adaptive history; threefold is not an untouched holdout. No public-score tuning.')
 dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])

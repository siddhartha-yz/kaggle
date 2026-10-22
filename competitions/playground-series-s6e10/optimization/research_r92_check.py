"""Complete-stage-only scoring, including a no-op reconstruction control."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import sys,json,hashlib,numpy as np
from sklearn.metrics import roc_auc_score
from research_r92_common import *
from research_r4_common import evidence

def main(stage):
 p=plan();tree,ce,base=refs(stage);K=5 if stage=='fivefold' else 3;pred=np.full(len(base['id']),np.nan);legacy=np.full(len(pred),np.nan)
 for k in range(K):
  run=OUT/stage/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r92_plan.json') and r['new_neural_fits']==0 and not r['official_validation_labels_used_for_calibration'] and r['native_members']==32 and r['origins_count']==32 and r['slope_fits']==r['weight_fits']==0
  for n,h in r['files_sha256'].items():assert sha(run/n)==h
  fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);o=json.loads((run/'origins.json').read_text());assert o['training_rows']==len(fi) and o['training_prior']==float(base['target'][fi].mean()) and o['training_indices_sha256']==hashlib.sha256(fi.tobytes()).hexdigest()
  assert len(o['bias'])==32 and not o['validation_labels_used'] and o['moment_error']<=1e-10
  z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],base['id'][vi]);pred[vi]=z['pred'];legacy[vi]=z['legacy_replay']
 assert np.isfinite(pred).all() and np.isfinite(legacy).all()
 noop=candidate(ce['candidate'],legacy);noop_delta=float(roc_auc_score(base['target'],noop)-roc_auc_score(base['target'],base['candidate']));assert abs(noop_delta)<=1e-7,'No-opcontrolAUCdrift;stopbeforetreatmentquality'
 treatment=candidate(ce['candidate'],pred);write_guard(40);np.savez_compressed(OUT/stage/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,legacy_replay=legacy,candidate=treatment,reference_v12=base['candidate'])
 ev=evidence(base['target'],base['candidate'],treatment,base['fold']);r=dict(status='complete',completed_utc=now(),stage=stage,plan_sha256=sha(ROOT/'research_r92_plan.json'),all_folds_completed_before_quality=True,no_op_control_delta_AUC=noop_delta,eligible=ev['eligible_for_new_partition_replication'],vs_actual_v12=ev,oof_sha256=sha(OUT/stage/'oof.npz'),new_neural_fits=0,new_formal_submissions=0)
 dump(OUT/stage/'result.json',r);print('STAGE_RESULT',json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])

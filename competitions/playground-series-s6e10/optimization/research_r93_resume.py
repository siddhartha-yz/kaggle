"""Recover frozen diagnosis, preserving original failed attempt and all completed stages."""
from research_r93_repair_common import *
from research_r93_driver import operation

def main():
 p,repair=repaired_plan();plan(all_inputs=True);reserve();OUT.mkdir(exist_ok=True)
 initial=ROOT/'research_r93_initial_attempt.json'
 if not initial.exists():
  assert json.loads((ROOT/'research_r93_failure.json').read_text())['status']=='stopped_for_assessment'
  dump(initial,dict(initial_state=json.loads((ROOT/'research_r93_result.json').read_text()),initial_failure=json.loads((ROOT/'research_r93_failure.json').read_text()),initial_state_sha256=sha(ROOT/'research_r93_result.json'),initial_failure_sha256=sha(ROOT/'research_r93_failure.json')))
 r=dict(status='resuming_frozen_basic_audit',started_utc=now(),pid=os.getpid(),plan_sha256=sha(ROOT/'research_r93_plan.json'),repair_sha256=sha(ROOT/'research_r93_repair_manifest.json'),new_SAT_classifier_fits=0,new_submissions=0,original_failure_preserved=initial.name)
 def state():dump(ROOT/'research_r93_result.json',r);dump(ROOT/'research_r93_recovery_result.json',r)
 state()
 if not (OUT/'basic.json').exists():operation('research_r93_basic_resumable.py',180)
 else:assert json.loads((OUT/'basic.json').read_text())['status']=='complete_covariates_only';print('SKIP_COMPLETED_BASIC',flush=True)
 r.update(status='resuming_fixed_domain_budget',updated_utc=now());state()
 if not (OUT/'adversarial.json').exists():operation('research_r93_av_resumable.py',300)
 else:assert json.loads((OUT/'adversarial.json').read_text())['status']=='complete_domain_diagnostic';print('SKIP_COMPLETED_DOMAIN_DIAGNOSTIC',flush=True)
 r.update(status='saved_domain_model_verification',updated_utc=now());state()
 if not (OUT/'integrity.json').exists():operation('research_r93_verify.py',120)
 else:assert json.loads((OUT/'integrity.json').read_text())['status']=='passed_diagnostic_integrity';print('SKIP_COMPLETED_INTEGRITY',flush=True)
 r.update(status='completed_diagnostic_pending_interpretation',completed_utc=now(),basic_sha256=sha(OUT/'basic.json'),adversarial_sha256=sha(OUT/'adversarial.json'),integrity_sha256=sha(OUT/'integrity.json'),artifact_bytes=ownbytes());state()
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r93_recovery_failure.json',dict(status='stopped_for_assessment',utc=now(),error=repr(e)));raise

from research_r93_common import *
import subprocess,sys,signal
from research_resume_policy_20261021 import check_training

def operation(script,timeout):
 check_training();reserve();child=subprocess.Popen([sys.executable,'-B','-u',script],cwd=ROOT,start_new_session=True);begin=time.monotonic()
 try:
  while child.poll() is None:
   check_training();reserve();assert psutil.virtual_memory().available>=4*2**30;assert time.monotonic()-begin<timeout,'Diagnosticoperationbudgetcheckpoint';time.sleep(2)
  assert child.returncode==0,f'{script}exit{child.returncode}'
 except BaseException:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
  raise

def main():
 p=plan(all_inputs=True);assert not (ROOT/'research_r93_result.json').exists();reserve();OUT.mkdir(exist_ok=True)
 result=dict(status='basic_covariate_audit',started_utc=now(),pid=os.getpid(),plan_sha256=sha(ROOT/'research_r93_plan.json'),new_SAT_classifier_fits=0,new_submissions=0)
 dump(ROOT/'research_r93_result.json',result);operation('research_r93_basic.py',180)
 result.update(status='fixed_budget_domain_diagnostic',updated_utc=now());dump(ROOT/'research_r93_result.json',result);operation('research_r93_av.py',300)
 result.update(status='diagnostic_integrity_check',updated_utc=now());dump(ROOT/'research_r93_result.json',result);operation('research_r93_verify.py',120)
 result.update(status='completed_diagnostic_pending_interpretation',completed_utc=now(),basic_sha256=sha(OUT/'basic.json'),adversarial_sha256=sha(OUT/'adversarial.json'),integrity_sha256=sha(OUT/'integrity.json'),artifact_bytes=ownbytes());dump(ROOT/'research_r93_result.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r93_failure.json',dict(status='stopped_for_assessment',utc=now(),error=repr(e)));raise

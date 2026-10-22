"""Probe first; inference/calibration only; all-fold gates with no automatic rescue."""
import os,sys,time,json,signal,subprocess,psutil
from research_r92_common import *
from research_resume_policy_20261021 import check_training

def operation(script,args,timeout):
 check_training();write_guard();assert psutil.virtual_memory().available>=6*2**30
 child=subprocess.Popen([sys.executable,'-B','-u',script,*args],cwd=ROOT,start_new_session=True);begin=time.monotonic()
 try:
  while child.poll() is None:
   check_training();write_guard();assert psutil.virtual_memory().available>=4*2**30
   proc=psutil.Process(child.pid);rss=sum(t.memory_info().rss for t in [proc]+proc.children(recursive=True) if t.is_running());assert rss<=8*2**30
   assert time.monotonic()-begin<timeout,'Boundedoperationcheckpoint;notcampaignstop'
   time.sleep(2)
  assert child.returncode==0,f'{script}exited{child.returncode}'
 except BaseException:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
  raise

def main():
 p=plan(all_inputs=True);assert not (ROOT/'research_r92_result.json').exists(),'Noautomaticrestart';write_guard();OUT.mkdir(exist_ok=True)
 result=dict(status='probe_running',started_utc=now(),pid=os.getpid(),plan_sha256=sha(ROOT/'research_r92_plan.json'),stages={},new_neural_fits=0,new_formal_submissions=0,actual_top500_verified=False)
 dump(ROOT/'research_r92_result.json',result);operation('research_r92_probe.py',[],300);probe=json.loads((OUT/'probe/result.json').read_text());assert probe['status']=='passed_no_quality_score'
 result['probe']=probe;dump(ROOT/'research_r92_result.json',result)
 if probe['estimated_fivefold_seconds']>1800:
  result.update(status='probe_passed_budget_checkpoint',completed_utc=now(),reason='Measuredfivefoldestimateabove30min;assessbeforenewstage');dump(ROOT/'research_r92_result.json',result);return
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   result.update(status='native_inference_and_origin_calibration',current_stage=stage,current_fold=k,updated_utc=now());dump(ROOT/'research_r92_result.json',result);operation('research_r92_worker.py',[stage,str(k)],1200)
  result.update(status='all_folds_complete_evaluating',current_stage=stage,updated_utc=now());dump(ROOT/'research_r92_result.json',result);operation('research_r92_check.py',[stage],600)
  stage_result=json.loads((OUT/stage/'result.json').read_text());result['stages'][stage]=stage_result;dump(ROOT/'research_r92_result.json',result)
  if not stage_result['eligible']:
   result.update(status='closed_not_qualified',completed_utc=now(),reason=f'{stage}failedoriginalfrozengate;norescue');dump(ROOT/'research_r92_result.json',result);return
 result.update(status='both_partitions_passed_requires_independent_audit',completed_utc=now());dump(ROOT/'research_r92_result.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r92_failure.json',dict(status='stopped_for_assessment',utc=now(),error=repr(e)));raise

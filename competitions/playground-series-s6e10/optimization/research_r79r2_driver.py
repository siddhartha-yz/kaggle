"""Bounded single-candidate validation; stage budget is not a campaign deadline."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess,time,signal
from research_resume_policy_20261021 import check_training,StopByDeadline
ROOT=Path(__file__).resolve().parent
def state(status,**kw):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**kw);p=ROOT/'research_r79r2_driver_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p);print(json.dumps(d),flush=True)
def run(script,args,log,timeout):
 with open(ROOT/log,'a') as f:
  p=subprocess.Popen([sys.executable,'-B','-u',script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);begin=time.monotonic()
  try:
   while p.poll() is None:
    check_training()
    if time.monotonic()-begin>=timeout:raise StopByDeadline('Boundedstageoperationbudgetexhausted;assessmentcheckpointonly')
    time.sleep(.5)
  except BaseException:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.kill();p.wait()
   raise
  assert p.returncode==0,f'{script} exited {p.returncode}'
def main():
 check_training();old=ROOT/'research_r79r2_driver_state.json'
 if old.exists() and json.loads(old.read_text())['status'] in ['not_qualified','qualified_requires_independent_audit','stopped_for_assessment']:
  print('NO_AUTOMATIC_RESTART_OF_TERMINAL_STAGE',flush=True);return
 assert json.loads((ROOT/'research_r79_trace_probe.json').read_text())['status']=='passed_no_official_quality_score'
 start=time.monotonic()
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   remaining=21600-(time.monotonic()-start)
   if remaining<=0:raise StopByDeadline('Sixhourstagebudgetexhausted;notacampaignstopcondition')
   state('native_GPU_fit_and_queries',stage=stage,fold=k)
   run('research_r79r2_worker.py',[stage,str(k)],f'research_r79r2_{stage}_fold{k}.log',min(3600,remaining))
   assert (ROOT/f'research_artifacts/r79r2_tabstar_raw21_{stage}/fold{k}/result.json').exists(),'Incompletefoldneverqualifies'
  state('all_folds_complete_evaluating',stage=stage)
  run('research_r79r2_check.py',[stage],f'research_r79r2_{stage}_check.log',600)
  if not json.loads((ROOT/f'research_artifacts/r79r2_tabstar_raw21_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage);return
 state('qualified_requires_independent_audit',stage='threefold')
if __name__=='__main__':
 def terminate(signum,frame):raise StopByDeadline('NormalSIGTERMforthistaskonly')
 signal.signal(signal.SIGTERM,terminate)
 try:main()
 except StopByDeadline as error:state('stopped_for_assessment',reason=str(error));sys.exit(0)
 except BaseException as error:state('failed',error_type=type(error).__name__,error=str(error)[:600]);raise

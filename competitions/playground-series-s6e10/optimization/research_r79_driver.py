"""Immediate serial fivefold then conditional threefold; total six-hour budget."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess,time,signal
from research_run_deadline import SOFT,HARD,utc_seconds,check_training,check_finalization,StopByDeadline
ROOT=Path(__file__).resolve().parent
def state(status,**kw):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**kw);p=ROOT/'research_r79_driver_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p);print(json.dumps(d),flush=True)
def main():
 check_training()
 old=ROOT/'research_r79_driver_state.json'
 if old.exists() and json.loads(old.read_text())['status'] in ['not_qualified','qualified_requires_independent_audit','stopped_by_user_deadline','stopped_insufficient_time_before_threefold']:
  print('NO_AUTOMATIC_RESUME_OF_TERMINAL_RUN',flush=True);return
 assert json.loads((ROOT/'research_r77_result.json').read_text())['status']=='not_qualified'
 assert json.loads((ROOT/'research_r79_trace_probe.json').read_text())['status']=='passed_no_official_quality_score'
 start=time.monotonic()
 for stage,K in [('fivefold',5),('threefold',3)]:
  if stage=='threefold':
   observed=[json.loads(p.read_text())['seconds'] for p in (ROOT/'research_artifacts/r79_tabstar_raw21_fivefold').glob('fold*/result.json')]
   if utc_seconds()+3*max(observed)+300>=SOFT:state('stopped_insufficient_time_before_threefold',remaining_seconds=SOFT-utc_seconds(),conservative_required_seconds=3*max(observed)+300);return
  for k in range(K):
   check_training()
   assert time.monotonic()-start<21600,'whole-run budget'
   state('native_GPU_fit_and_queries',stage=stage,fold=k)
   with open(ROOT/f'research_r79_{stage}_fold{k}.log','a') as f:
    p=subprocess.Popen([sys.executable,'-B','-u','research_r79_worker.py',stage,str(k)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);begin=time.monotonic()
    try:
     while p.poll() is None:
      if utc_seconds()>=SOFT or time.monotonic()-begin>=3600:raise StopByDeadline('Foldbudgetor15:15UTCtrainingcutoff')
      time.sleep(.5)
    except BaseException:
     if p.poll() is None:
      p.terminate()
      try:p.wait(timeout=20)
      except subprocess.TimeoutExpired:p.kill();p.wait()
     raise
    assert p.returncode==0,f'worker exited {p.returncode}'
    assert (ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/fold{k}/result.json').exists(),'Workerstoppedwithoutcompletefold;neverqualify'
  state('all_folds_complete_evaluating',stage=stage)
  with open(ROOT/f'research_r79_{stage}_check.log','a') as f:subprocess.run([sys.executable,'-B','-u','research_r79_check.py',stage],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  if not json.loads((ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage);return
 state('qualified_requires_independent_audit',stage='threefold')
if __name__=='__main__':
 def terminate(signum,frame):raise StopByDeadline('NormalSIGTERM;onlythisservicechildrenareterminated')
 signal.signal(signal.SIGTERM,terminate)
 try:main()
 except StopByDeadline as error:state('stopped_by_user_deadline',reason=str(error));sys.exit(0)
 except BaseException as error:state('failed',error_type=type(error).__name__,error=str(error)[:600]);raise

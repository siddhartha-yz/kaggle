"""Serial protected pair execution; bounded90min experiment, never a campaign deadline."""
from research_r94_common import *
from research_resume_policy_20261021 import check_training
import subprocess,sys,time,signal,fcntl

def state(status,**kw):
 d=dict(status=status,updated_utc=now(),PID=os.getpid(),actual_top500_verified=False,**kw)
 dump(ROOT/'research_r94_driver_state.json',d);print(json.dumps(d),flush=True)
def run(script,args,limit):
 log=ROOT/('research_r94_'+('_'.join(args) if args else 'operation')+'_'+Path(script).stem+'.log')
 with log.open('a') as f:
  p=subprocess.Popen([sys.executable,'-B','-u',script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  tick=time.monotonic();state('child_running',script=script,args=args,child_PID=p.pid,log=str(log),timeout_seconds=limit)
  try:
   while p.poll() is None:
    check_training();reserve()
    if time.monotonic()-tick>=limit:raise TimeoutError('Frozen experiment operation budget reached; checkpoint preserved')
    time.sleep(1)
  except BaseException:
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
   raise
  assert p.returncode==0,f'{script} exited {p.returncode}; inspect {log}'
def main():
 plan(inputs=True);check_training();reserve();OUT.mkdir(exist_ok=True)
 lock=(OUT/'controller.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('EXISTING_CONTROLLER_OWNS_EXPERIMENT; no duplicate start',flush=True);return
 terminal=ROOT/'research_r94_driver_state.json'
 if terminal.exists():
  old=json.loads(terminal.read_text())
  assert old['status'] not in ['closed_not_qualified','qualified_pending_independent_audit','stopped_for_assessment'],'No automatic restart of terminal experiment'
 start=time.monotonic()
 for stage,K in [('fivefold',5),('threefold',3)]:
  if stage=='threefold':assert json.loads((OUT/'fivefold/result.json').read_text())['eligible']
  for k in range(K):
   remaining=5400-(time.monotonic()-start);assert remaining>0
   run('research_r94_worker.py',[stage,str(k)],min(900,remaining))
  remaining=5400-(time.monotonic()-start);assert remaining>0
  run('research_r94_check.py',[stage],min(180,remaining))
  result=json.loads((OUT/stage/'result.json').read_text())
  if not result['eligible']:
   state('closed_not_qualified',stage=stage,result_sha256=sha(OUT/stage/'result.json'),new_submissions=0,elapsed_seconds=time.monotonic()-start);return
 state('qualified_pending_independent_audit',elapsed_seconds=time.monotonic()-start,new_submissions=0)
if __name__=='__main__':
 try:main()
 except BaseException as error:state('stopped_for_assessment',error_type=type(error).__name__,reason=str(error));raise

"""Single fixed candidate, bounded fits, durable fold reuse, conditional replication."""
from research_r79r2_driver import ROOT,run,StopByDeadline,check_training
from datetime import datetime,timezone
import json,os,time,sys,signal
def state(status,**kw):
 r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),actual_top500_verified=False,**kw);p=ROOT/'research_r78_driver_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p);print(json.dumps(r),flush=True)
def main():
 check_training();p=ROOT/'research_r78_driver_state.json'
 if p.exists() and json.loads(p.read_text())['status'] in ['not_qualified','qualified_requires_independent_audit','stopped_for_assessment','failed']:raise RuntimeError('Terminalstagealreadyexists;assessbeforeanyretry')
 tick=time.monotonic()
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   assert time.monotonic()-tick<7200,'Twohourboundedstagecheckpoint;notcampaignstop';check_training();state('native_GPU_fit_and_queries',stage=stage,fold=k);run('research_r78_worker.py',[stage,str(k)],f'research_r78_{stage}_fold{k}.log',900)
  state('all_folds_complete_evaluating',stage=stage);run('research_r78_check.py',[stage],f'research_r78_{stage}_check.log',600)
  if not json.loads((ROOT/f'research_artifacts/r78_logratio_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage);return
 state('qualified_requires_independent_audit',stage='threefold')
if __name__=='__main__':
 def terminate(signum,frame):raise StopByDeadline('NormalSIGTERMforthistaskonly')
 signal.signal(signal.SIGTERM,terminate)
 try:main()
 except StopByDeadline as e:state('stopped_for_assessment',reason=str(e));sys.exit(0)
 except BaseException as e:state('failed',error_type=type(e).__name__,error=str(e)[:600]);raise

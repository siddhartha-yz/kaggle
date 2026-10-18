"""Serial recovery driver; waiting is bounded by identified baseline process lifetime."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,time,json,subprocess,argparse
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'.venv-gpu/bin/python'
def dump(status,**kw):
 p=ROOT/'research_r4_driver_state.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'status':status,'driver_pid':os.getpid(),**kw},indent=2)+'\n');temp.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline-pid',type=int,required=True);a=ap.parse_args();dump('waiting_for_existing_baseline',baseline_pid=a.baseline_pid)
 baseline=ROOT/'research_artifacts/r4_baseline_replication3/result.json'
 while True:
  try:
   r=json.loads(baseline.read_text())
   if r['status']=='complete':break
  except (FileNotFoundError,json.JSONDecodeError):pass
  try:os.kill(a.baseline_pid,0)
  except ProcessLookupError:raise RuntimeError('Baseline process ended without complete result; inspectbaseline log')
  time.sleep(5)
 for runner,log,result in [('research_r4_lean_replication3.py','research_r4_lean_replication3.log','research_artifacts/r4_lean_gpu_init_replication3/result.json'),('research_r4_csv.py','research_r4_csv.log','research_artifacts/r4_lean_cv_final/result.json')]:
  target=ROOT/result
  if target.exists() and json.loads(target.read_text()).get('status')=='complete':r=json.loads(target.read_text())
  else:
   dump('running_serial_stage',runner=runner,log=log)
   with open(ROOT/log,'a') as f:p=subprocess.run([str(PYTHON),runner],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   if p.returncode:raise RuntimeError(f'{runner} exited{p.returncode};inspect{log}')
   r=json.loads(target.read_text());assert r['status']=='complete'
  if runner=='research_r4_lean_replication3.py' and not r['eligible_for_new_partition_replication']:
   dump('replication_did_not_qualify_no_csv',result=result,whole_top100_task_completed=False);return
 dump('candidate_csv_ready_for_library_and_parent',result=result,whole_top100_task_completed=False,new_formal_used=0)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump('failed',error=repr(e),whole_top100_task_completed=False);raise

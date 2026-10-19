"""Bounded serial continuation after ExtraTrees, with no duplicate baseline fits."""
from research_w3_common import *
import time,subprocess,psutil
STAMP=ROOT/'research_w3_nested_neural_queue_state.json'
def launch(split,state):
 start_guard();args=[str(ROOT.parent/'.venv-gpu/bin/python'),'-u',str(ROOT/'research_w3_nested_neural.py'),'--split',split];deadline=datetime.fromisoformat(plan()['limits']['training_hard_cutoff_utc']);remaining=(deadline-datetime.now(timezone.utc)).total_seconds();assert remaining>120;proc=None;log=ROOT/f'research_w3_nested_neural_{split}.log'
 with log.open('a') as f:
  try:
   proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);state.update(stage=split,child_pid=proc.pid,current_command=args,last_progress_utc=now(),active_log=str(log));dump(STAMP,state);code=proc.wait(timeout=min(5400 if split=='oof' else 1800,remaining-2));assert code==0,f'NestedNN stage{split} failed code={code}; see{log}'
  finally:
   if proc is not None and proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
def main():
 assert not STAMP.exists(),'Queue already registered: inspect actual process/state before any restart';state={'status':'running','stage':'waiting_for_extra_trees_primary','started_utc':now(),'queue_pid':os.getpid(),'nested_plan_sha256':sha(ROOT/'research_w3_nested_neural_plan.json'),'completed_stages':[]};dump(STAMP,state);started=time.monotonic()
 try:
  extra=ROOT/'research_artifacts/w3_extra_trees_primary/result.json'
  while True:
   if extra.exists():
    latest=json.loads(extra.read_text())
    if latest.get('status')=='complete':break
    assert latest.get('status')!='failed','ExtraTrees failed; retain its logs and do not launch a new model automatically'
   assert datetime.now(timezone.utc)<datetime.fromisoformat(plan()['limits']['training_start_cutoff_utc']);assert time.monotonic()-started<1800,'ExtraTrees result wait exceeded30min';active=[]
   for p in psutil.process_iter(['pid','cmdline']):
    try:
     cmd=p.info['cmdline'] or []
     if any(Path(x).name=='research_w3_extra_trees.py' for x in cmd) and 'primary' in cmd:active.append(p.info['pid'])
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   if not active and extra.exists() and json.loads(extra.read_text()).get('status')=='complete':break
   assert active,'ExtraTrees process absent without a completed result; do not wait indefinitely';state.update(extra_trees_pids=active,last_progress_utc=now());dump(STAMP,state);time.sleep(15)
  er=json.loads(extra.read_text());assert er['status']=='complete'
  if er['eligible_for_next']:state.update(status='complete',outcome='ExtraTrees passed; prioritize its confirmation before nestedNN',completed_utc=now());dump(STAMP,state);return
  for split in ['probe','primary','confirm','oof']:
   launch(split,state);state['completed_stages'].append(split);state['child_pid']=None;dump(STAMP,state)
   if split=='probe':assert json.loads((ROOT/'research_w3_nested_neural_hardware.json').read_text())['eligible_resources'];continue
   result=json.loads((ROOT/f'research_artifacts/w3_nested_neural_{split}/result.json').read_text())
   if not result['eligible_for_next']:state.update(status='complete',outcome=f'{split}_gate_failed',completed_utc=now());dump(STAMP,state);return
  state.update(status='complete',outcome='OOF_qualified_ready_for_full_meta_fit_and_CSV_audit',completed_utc=now());dump(STAMP,state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(STAMP,state);raise
if __name__=='__main__':main()

"""Wait for preceding serial phase; evaluate independent-unlabeled profile recipe without overlap."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,subprocess,sys
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def main():
 state={'status':'waiting_for_phase2','started_utc':now()};dump(ROOT/'research_w3_phase3_state.json',state)
 try:
  while True:
   assert datetime.now(timezone.utc)<datetime.fromisoformat(read(ROOT/'research_w3_plan.json')['limits']['training_start_cutoff_utc']);p=read(ROOT/'research_w3_phase2_state.json')
   if p['status']=='complete':break
   if p['status']=='failed':raise RuntimeError('Previousphasefailed;requiresinspectionbeforecontinuing')
   time.sleep(10)
  if read(ROOT/'research_w3_phase2_selection.json')['qualified_for_full']:state.update(status='waiting_for_qualified_full_priority',ended_utc=now());dump(ROOT/'research_w3_phase3_state.json',state);return
  selection={'status':'validating','eligible_for_full':False,'results':{}}
  for split in ['primary','confirm','oof']:
   folder=ROOT/'research_artifacts'/f'w3_external_tree_{split}';result=folder/'result.json'
   if result.exists():r=read(result);assert r['status']=='complete'
   else:
    assert not folder.exists();state.update(status='training',active=split,updated_utc=now());dump(ROOT/'research_w3_phase3_state.json',state)
    with (ROOT/f'research_w3_external_tree_{split}.log').open('x') as log:
     proc=subprocess.Popen([sys.executable,'-u','research_w3_external_tree.py','--split',split],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state['child_pid']=proc.pid;dump(ROOT/'research_w3_phase3_state.json',state);rc=proc.wait()
    assert rc==0 and result.exists();r=read(result)
   selection['results'][split]=r;dump(ROOT/'research_w3_phase3_selection.json',selection)
   if not r['eligible_for_next']:selection.update(status=split+'_rejected',completed_utc=now());dump(ROOT/'research_w3_phase3_selection.json',selection);state.update(status='complete',active=None,outcome=split+'_rejected',completed_utc=now());dump(ROOT/'research_w3_phase3_state.json',state);return
  selection.update(status='qualified_for_full',eligible_for_full=True,completed_utc=now());dump(ROOT/'research_w3_phase3_selection.json',selection);state.update(status='complete',active=None,outcome='qualified_for_full',completed_utc=now());dump(ROOT/'research_w3_phase3_state.json',state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(ROOT/'research_w3_phase3_state.json',state);raise
if __name__=='__main__':main()

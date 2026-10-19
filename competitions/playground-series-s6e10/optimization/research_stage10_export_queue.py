"""Wait for nested OOF, then export only when its unchanged gate passes."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
state=ROOT/'research_stage10_export_state.json';protocol=json.loads((ROOT/'research_protocol.json').read_text());deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
save('waiting_for_nested_validation','waiting')
while True:
 assert datetime.now(timezone.utc)<deadline
 previous=json.loads((ROOT/'research_stage9_queue_state.json').read_text())
 if previous['status']=='complete':break
 if previous['status']=='failed':save('upstream_failed','failed',upstream=previous);raise RuntimeError(previous)
 time.sleep(5)
selection=json.loads((ROOT/'research_nested_stack_selection.json').read_text())
if not selection.get('oof',{}).get('eligible',False):save('no_eligible_candidate','complete');sys.exit(0)
assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc']);save('nested_full_refit','running')
try:
 with (ROOT/'research_nested_full.log').open('w') as f:subprocess.run([sys.executable,'-u','research_nested_full.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
except BaseException as e:save('nested_full_refit','failed',error=repr(e));raise
save('verified_csv_ready','complete',audit='submission_nested_stack_v5_audit.json')

"""Serialize a gated deeper-CB export after the validation queue completes."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
state=ROOT/'research_stage13_export_state.json';protocol=json.loads((ROOT/'research_protocol.json').read_text());deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
save('waiting_for_cb_deep_validation','waiting')
while True:
 assert datetime.now(timezone.utc)<deadline
 previous=json.loads((ROOT/'research_stage12_queue_state.json').read_text())
 if previous['status']=='complete':break
 if previous['status']=='failed':save('upstream_failed','failed',upstream=previous);raise RuntimeError(previous)
 time.sleep(5)
selection=json.loads((ROOT/'research_cb_deep_selection.json').read_text());w=selection.get('chosen_weight');r=selection.get('oof',{}).get('candidates',{}).get(str(w),{})
if not r.get('eligible',False):save('no_eligible_candidate','complete');sys.exit(0)
elapsed=json.loads((ROOT/'research_artifacts/cb_deep_confirm/fold0/result.json').read_text())['elapsed_seconds'];required=1.4*elapsed+(600 if w<1 else 180)
if (deadline-datetime.now(timezone.utc)).total_seconds()<required:save('insufficient_refit_budget','complete',estimated_seconds_needed=required);sys.exit(0)
assert datetime.now(timezone.utc)<datetime.fromisoformat(protocol['training_start_cutoff_utc']);save('cb_deep_full_refit','running')
try:
 with (ROOT/'research_cb_deep_full.log').open('w') as f:subprocess.run([sys.executable,'-u','research_cb_deep_full.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
except BaseException as e:save('cb_deep_full_refit','failed',error=repr(e));raise
save('verified_csv_ready','complete',audit='submission_cb_deep_v5_audit.json')

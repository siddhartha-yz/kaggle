"""Wait for validation completion; never fit unless fixed OOF gates and budget pass."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;plan=json.loads((ROOT/'research_realmlp_w2_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc']);state=ROOT/'research_w2_export_state.json'
def save(status,**extra):state.write_text(json.dumps(dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n')
save('waiting_for_validation')
while True:
 p=ROOT/'research_stage16_w2_state.json'
 if p.exists():
  r=json.loads(p.read_text())
  if r['status']=='failed':save('validation_failed');sys.exit(1)
  if r['status']=='complete':break
 if (deadline-datetime.now(timezone.utc)).total_seconds()<600:save('budget_expired');sys.exit(0)
 time.sleep(10)
selection=json.loads((ROOT/'research_realmlp_w2_selection.json').read_text())
if not selection['eligible_for_full']:save('complete',stage='no_eligible_candidate');sys.exit(0)
times=[json.loads(p.read_text())['fit_seconds'] for p in (ROOT/'research_artifacts/realmlp256_w2_oof').glob('fold*/result.json')];required=max(times)*1.25*1.02+90;remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
if required>remaining:save('complete',stage='insufficient_full_fit_budget',required_seconds=required,remaining_seconds=remaining);sys.exit(0)
save('running',stage='full_refit',required_seconds=required,remaining_seconds=remaining)
try:
 with (ROOT/'research_realmlp_w2_full.log').open('w') as f:subprocess.run([sys.executable,'-u','research_realmlp_w2_full.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
except BaseException as e:save('failed',error=repr(e));raise
save('complete',stage='audited_candidate_ready',audit='submission_realmlp256_v6_audit.json')

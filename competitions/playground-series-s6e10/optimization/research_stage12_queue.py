"""Bound one deeper CB hypothesis; no duplicated fits, fixed validation gates."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'research_stage11_queue_state.json').read_text())['status']=='complete'
state=ROOT/'research_stage12_queue_state.json';deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc'])
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def run(split,trees=None,weight=None):
 save('cb_deep_'+split,'running');args=['research_cb_deep.py','--split',split]
 if trees is not None:args+=['--trees',str(trees),'--weight',str(weight)]
 timeout=(deadline-datetime.now(timezone.utc)).total_seconds()
 if split=='primary':timeout=min(timeout,1500)
 print('START',split,flush=True)
 try:
  with (ROOT/f'research_cb_deep_{split}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
 except BaseException as e:save('cb_deep_'+split,'failed',error=repr(e));raise
 print('DONE',split,flush=True)
run('primary');primary=json.loads((ROOT/'research_artifacts/cb_deep_primary/result.json').read_text());scores=primary['candidates'];eligible=[w for w,r in scores.items() if r['vs_v4']['delta_auc']>=.0001 and r['vs_v4']['conditional_95pct_ci'][0]>0 and r['vs_core']['delta_auc']>0];selection={'primary':primary,'chosen_weight':None,'public_scores_used':False}
if eligible:
 w=max(eligible,key=lambda w:scores[w]['vs_v4']['candidate_auc']);trees=int(primary['selected_trees'][0]);selection.update(chosen_weight=float(w),trees=trees,frozen_utc=datetime.now(timezone.utc).isoformat());(ROOT/'research_cb_deep_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 elapsed=json.loads((ROOT/'research_artifacts/cb_deep_primary/fold0/result.json').read_text())['elapsed_seconds'];required=6*elapsed*1.2+1200
 if (deadline-datetime.now(timezone.utc)).total_seconds()<required:selection['blocked_by_budget']={'estimated_seconds_needed':required}
 else:
  run('confirm',trees,float(w));confirmation=json.loads((ROOT/'research_artifacts/cb_deep_confirm/result.json').read_text());selection['confirmation']=confirmation;c=confirmation['candidates'][w]
  if c['vs_v4']['conditional_95pct_ci'][0]>0 and c['vs_core']['delta_auc']>0:
   elapsed=json.loads((ROOT/'research_artifacts/cb_deep_confirm/fold0/result.json').read_text())['elapsed_seconds'];required=5*elapsed*1.2+1200
   if (deadline-datetime.now(timezone.utc)).total_seconds()<required:selection['blocked_by_budget']={'estimated_seconds_needed':required}
   else:run('oof',trees,float(w));selection['oof']=json.loads((ROOT/'research_artifacts/cb_deep_oof/result.json').read_text())
(ROOT/'research_cb_deep_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete')

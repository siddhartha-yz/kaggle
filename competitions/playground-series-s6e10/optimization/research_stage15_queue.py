"""Continue the already-running fixed RealMLP primary only if frozen gates pass."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,subprocess,sys
ROOT=Path(__file__).resolve().parent;plan=json.loads((ROOT/'research_realmlp_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc']);state=ROOT/'research_stage15_queue_state.json'
def save(stage,status,**kw):state.write_text(json.dumps(dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n')
def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
def run(split):
 save(split,'running');print('START',split,flush=True)
 try:
  with (ROOT/f'research_realmlp_{split}.log').open('w') as f:subprocess.run([sys.executable,'-u','research_realmlp.py','--split',split],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining()),check=True)
 except BaseException as e:save(split,'failed',error=repr(e));raise
 print('DONE',split,flush=True);return json.loads((ROOT/'research_artifacts'/f'realmlp48_{split}'/'result.json').read_text())
save('existing_primary','waiting');p=ROOT/'research_artifacts/realmlp48_primary/result.json'
while not p.exists():
 r=ROOT/'research_artifacts/realmlp48_primary/fold0/result.json'
 if r.exists() and json.loads(r.read_text()).get('status')=='failed':save('primary','failed');sys.exit(1)
 if remaining()<900:save('primary','budget_expired');sys.exit(1)
 time.sleep(5)
selection={'primary':json.loads(p.read_text()),'eligible_for_full':False,'public_scores_used':False,'reference':'v5','weights':{'v5':.75,'realmlp':.25},'fixed_epochs':48}
c=selection['primary']['vs_v5'];r=json.loads((p.parent/'fold0/result.json').read_text());needed=r['elapsed_seconds']*7.25*1.3+600;selection['estimated_remaining_seconds']=needed
if c['delta_auc']>=plan['primary_gate']['minimum_delta_vs_v5'] and c['conditional_95pct_ci'][0]>0:
 if needed>remaining():selection['blocked_by_budget']=True
 else:
  selection['confirmation']=run('confirm');c=selection['confirmation']['vs_v5']
  if c['conditional_95pct_ci'][0]>0:
   selection['oof']=run('oof');o=selection['oof'];selection['eligible_for_full']=o['vs_v5']['delta_auc']>=plan['oof_gate']['minimum_pooled_delta_vs_v5'] and o['vs_v5']['conditional_95pct_ci'][0]>0 and all(c['delta_auc']>0 for c in o['fold_comparisons'])
(ROOT/'research_realmlp_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete',eligible_for_full=selection['eligible_for_full']);print(json.dumps(selection,indent=2),flush=True)

"""Window2: one256epoch RealMLP recipe, serial fits, frozen gates and deadline budget."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent;plan=json.loads((ROOT/'research_realmlp_w2_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc']);state=ROOT/'research_stage16_w2_state.json';selection_path=ROOT/'research_realmlp_w2_selection.json'
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def save(stage,status,**kw):dump(state,dict(stage=stage,status=status,utc=now(),**kw))
def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
def run(split):
 save(split,'running');print('START',split,now(),flush=True)
 try:
  with (ROOT/f'research_realmlp_w2_{split}.log').open('w') as f:subprocess.run([sys.executable,'-u','research_realmlp_w2.py','--split',split],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining()),check=True)
 except BaseException as e:save(split,'failed',error=repr(e));raise
 print('DONE',split,now(),flush=True);return json.loads((ROOT/'research_artifacts'/f'realmlp256_w2_{split}'/'result.json').read_text())
selection={'reference':'v5','weights':{'v5':.75,'realmlp':.25},'epochs':256,'public_scores_used':False,'eligible_for_full':False,'full_started':False,'started_utc':now()}
def budget(fit_seconds,equivalent_fits):
 required=fit_seconds*equivalent_fits*1.02+90
 selection['latest_runtime_budget']={'required_seconds':required,'available_seconds':remaining(),'equivalent_fits':equivalent_fits,'checked_utc':now()};dump(selection_path,selection)
 if required>remaining():selection['blocked_by_budget']=True;return False
 return True
selection['primary']=run('primary');p=selection['primary'];c=p['vs_v5'];fit=json.loads((ROOT/'research_artifacts/realmlp256_w2_primary/fold0/result.json').read_text())['fit_seconds'];dump(selection_path,selection)
if not p['futility_stop'] and p['actual_epochs']==[256] and c['delta_auc']>=.0001 and c['conditional_95pct_ci'][0]>0 and budget(fit,7.25):
 selection['confirmation']=run('confirm');dump(selection_path,selection);c=selection['confirmation']['vs_v5'];fit=json.loads((ROOT/'research_artifacts/realmlp256_w2_confirm/fold0/result.json').read_text())['fit_seconds']
 if c['conditional_95pct_ci'][0]>0 and budget(fit,6.25):
  selection['oof']=run('oof');dump(selection_path,selection);o=selection['oof']
  if not o.get('rejected_reason'):
   selection['eligible_for_full']=o['vs_v5']['delta_auc']>=.0002 and o['vs_v5']['conditional_95pct_ci'][0]>0 and all(c['delta_auc']>0 for c in o['fold_comparisons'])
selection['completed_utc']=now();dump(selection_path,selection);save('complete','complete',eligible_for_full=selection['eligible_for_full']);print(json.dumps(selection,indent=2),flush=True)

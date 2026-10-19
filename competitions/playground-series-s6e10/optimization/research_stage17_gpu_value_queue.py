"""Fixed GPU value-view tests, one confirmation and gated OOF/full; disk-backed logs."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parent;plan=json.loads((ROOT/'research_gpu_value_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc']);state=ROOT/'research_stage17_gpu_value_state.json';selection_path=ROOT/'research_gpu_value_selection.json'
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(stage,status,**kw):dump(state,dict(stage=stage,status=status,utc=now(),**kw))
def remaining():return (deadline-datetime.now(timezone.utc)).total_seconds()
def run(split,variant,n):
 name=f'gpu_value_{variant}_{split}_n{n}';out=ROOT/'research_artifacts'/name;result=out/'result.json'
 if result.exists():
  r=json.loads(result.read_text())
  if r['status']=='complete':
   old=json.loads((out/'plan.json').read_text());assert old['frozen_protocol']==plan and old['feature_sha256']==sha(ROOT/'research_gpu_value_features.py');return r
 save(name,'running');print('START',name,now(),flush=True)
 try:
  with (ROOT/f'research_{name}.log').open('w') as f:subprocess.run([sys.executable,'-u','research_gpu_value.py','--split',split,'--variant',variant,'--n-ens',str(n)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,remaining()))
 except BaseException as e:save(name,'failed',error=repr(e));raise
 return json.loads(result.read_text())
# Hardware selection before any new validation score, using only no-score probe.
n=None
for candidate in plan['hardware_selection']['n_ens_candidates']:
 try:r=run('probe','context',candidate)
 except subprocess.CalledProcessError:
  p=ROOT/'research_artifacts'/f'gpu_value_context_probe_n{candidate}'/'result.json'
  if p.exists() and json.loads(p.read_text()).get('hardware_oom'):continue
  raise
 if r['status']=='complete':n=candidate;break
assert n is not None,'No safe ensemble count fits the frozen GPU cap'
dump(ROOT/'research_gpu_value_hardware.json',{'frozen_utc':now(),'n_ens':n,'score_evaluated':False,'probe_name':f'gpu_value_context_probe_n{n}','peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes'],'cap_bytes':plan['gpu_allocator_cap_bytes']})
selection={'primary':{},'chosen_variant':None,'n_ens':n,'epochs':3,'weights':{'v5':.75,'realmlp':.25},'public_scores_used':False,'eligible_for_full':False,'started_utc':now()}
for variant in plan['variants']:selection['primary'][variant]=run('primary',variant,n);dump(selection_path,selection)
eligible=[v for v,r in selection['primary'].items() if r['vs_v5']['delta_auc']>=.0001 and r['vs_v5']['conditional_95pct_ci'][0]>0]
if eligible:
 v=max(eligible,key=lambda v:selection['primary'][v]['vs_v5']['candidate_auc']);selection.update(chosen_variant=v,view_frozen_utc=now());dump(selection_path,selection);selection['confirmation']=run('confirm',v,n);dump(selection_path,selection)
 if selection['confirmation']['vs_v5']['conditional_95pct_ci'][0]>0:
  selection['oof']=run('oof',v,n);o=selection['oof'];dump(selection_path,selection)
  if not o.get('rejected_reason'):selection['eligible_for_full']=o['vs_v5']['delta_auc']>=.0002 and o['vs_v5']['conditional_95pct_ci'][0]>0 and all(c['delta_auc']>0 for c in o['fold_comparisons'])
selection['validation_completed_utc']=now();dump(selection_path,selection)
if selection['eligible_for_full']:
 save('full_refit','running')
 try:
  with (ROOT/'research_gpu_value_full.log').open('w') as f:subprocess.run([sys.executable,'-u','research_gpu_value_full.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,remaining()))
 except BaseException as e:save('full_refit','failed',error=repr(e));raise
 selection['full_audit']='submission_realmlp_value_v6_audit.json'
selection['completed_utc']=now();dump(selection_path,selection);save('complete','complete',eligible_for_full=selection['eligible_for_full']);print(json.dumps(selection,indent=2),flush=True)

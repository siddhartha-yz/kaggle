"""Two frozen source-transfer controls, one confirmation and gated OOF."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'research_transfer_plan.json').read_text());deadline=datetime.fromisoformat(plan['limits']['training_hard_cutoff_utc']);state=ROOT/'research_stage14_queue_state.json'
def save(stage,status,**kw):state.write_text(json.dumps(dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n')
def run(split,p,epochs=None):
 name=f'nn_transfer_p{p}_{split}'+(f'_e{epochs}' if epochs is not None else '');save(name,'running');args=[sys.executable,'-u','research_transfer.py','--split',split,'--pretrain',str(p)]
 if epochs is not None:args+=['--epochs',str(epochs),'--fixed']
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,(deadline-datetime.now(timezone.utc)).total_seconds()))
 except BaseException as e:save(name,'failed',error=repr(e));raise
 print('DONE',name,flush=True);return json.loads((ROOT/'research_artifacts'/name/'result.json').read_text())
selection={'primary':{},'selected':None,'public_scores_used':False,'reference':'v5','final_gate_delta':.0002}
for p in plan['pretrain_epochs']:selection['primary'][str(p)]=run('primary',p)
selected=[p for p,r in selection['primary'].items() if r['vs_v5']['delta_auc']>=.0001 and r['vs_v5']['conditional_95pct_ci'][0]>0]
if selected:
 p=max(selected,key=lambda p:selection['primary'][p]['vs_v5']['candidate_auc']);epochs=selection['primary'][p]['selected_epochs'][0];selection.update(selected={'pretrain':int(p),'epochs':epochs,'weight':.25},frozen_utc=datetime.now(timezone.utc).isoformat());(ROOT/'research_transfer_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 r=json.loads((ROOT/'research_artifacts'/selection['primary'][p]['name']/'fold0/result.json').read_text());per_epoch=sum(e['training_seconds'] for e in r['epochs'])/len(r['epochs']);required=(per_epoch*epochs+r['preprocessing_seconds']+sum(e['training_seconds'] for e in r['pretrain_epochs']))*7*1.4+600
 if required>(deadline-datetime.now(timezone.utc)).total_seconds():selection['blocked_by_budget']={'estimated_seconds':required}
 else:
  selection['confirmation']=run('confirm',int(p),epochs)
  if selection['confirmation']['vs_v5']['conditional_95pct_ci'][0]>0:
   selection['oof']=run('oof',int(p),epochs);o=selection['oof'];selection['eligible_for_full']=o['vs_v5']['delta_auc']>=.0002 and o['vs_v5']['conditional_95pct_ci'][0]>0 and all(r['delta_auc']>0 for r in o['fold_comparisons'])
selection.setdefault('eligible_for_full',False);(ROOT/'research_transfer_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete',eligible_for_full=selection['eligible_for_full']);print(json.dumps(selection,indent=2),flush=True)

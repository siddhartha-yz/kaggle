"""Two source-context representations; promote only the primary winner."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'research_stage10_export_state.json').read_text())['status']=='complete'
state=ROOT/'research_stage11_queue_state.json';deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc'])
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def run(name,args):
 save(name,'running');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:save(name,'failed',error=repr(e));raise
 print('DONE',name,flush=True)
def path(mode,split):return ROOT/'research_artifacts'/f'lgb_stats_orig_{mode}_{split}'/'result.json'
def command(mode,split):return ['research_variant.py','--kind','lgb','--external',mode,'--split',split]
run('research_original_context',['research_original_context.py']);modes=json.loads((ROOT/'research_original_context_plan.json').read_text())['variants'];scores={}
for mode in modes:
 run('research_context_'+mode+'_primary',command(mode,'primary'));scores[mode]=json.loads(path(mode,'primary').read_text())['vs_original_lgb']
eligible=[mode for mode in modes if scores[mode]['delta_auc']>=.0001 and scores[mode]['conditional_95pct_ci'][0]>0];selection={'primary_scores':scores,'chosen':None,'public_scores_used':False}
if eligible:
 mode=max(eligible,key=lambda n:scores[n]['candidate_auc']);selection.update(chosen={'name':mode,'external':mode,'representation':'stats','profile':'base'},trees=639,frozen_utc=datetime.now(timezone.utc).isoformat());(ROOT/'research_context_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 run('research_context_confirm',command(mode,'confirm'));c=json.loads(path(mode,'confirm').read_text())['vs_original_lgb'];selection['confirmation']=c
 if c['conditional_95pct_ci'][0]>0:
  run('research_context_oof',command(mode,'oof'));selection['oof_result_path']=str(path(mode,'oof'))
(ROOT/'research_context_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete')

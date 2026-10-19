"""Four frozen information/regularization contrasts after current OOF; promote one."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc']);state=ROOT/'research_stage8_queue_state.json'
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
save('waiting_for_stage7','waiting')
while True:
 assert datetime.now(timezone.utc)<deadline
 previous=json.loads((ROOT/'research_stage7_queue_state.json').read_text())
 if previous['status']=='complete':break
 if previous['status']=='failed':save('waiting_for_stage7','failed',upstream=previous);raise RuntimeError(previous)
 time.sleep(5)
def run(name,args):
 save(name,'running');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:save(name,'failed',error=repr(e));raise
 print('DONE',name,flush=True)
def folder(v,split,trees=None,early=False):
 name=f"lgb_stats_orig_{v['external']}_{split}"
 if v['representation']!='stats':name+='_'+v['representation']
 if v['profile']=='wide':name+=f'_wide_t{trees}'+('_es' if early else '')
 return ROOT/'research_artifacts'/name
def command(v,split,trees=None,early=False):
 a=['research_variant.py','--kind','lgb','--external',v['external'],'--representation',v['representation'],'--profile',v['profile'],'--split',split]
 if trees is not None:a+=['--trees',str(trees)]
 if early:a+=['--early-stop']
 return a
run('research_original_xgb_teacher',['research_original_xgb.py'])
variants=json.loads((ROOT/'research_four_contrasts_plan.json').read_text())['variants'];scores={}
for v in variants:
 wide=v['profile']=='wide';run('research_screen_'+v['name'],command(v,'primary',3000 if wide else None,wide));r=json.loads((folder(v,'primary',3000 if wide else None,wide)/'result.json').read_text());scores[v['name']]=r['vs_original_lgb']
eligible=[v for v in variants if scores[v['name']]['delta_auc']>=.0001 and scores[v['name']]['conditional_95pct_ci'][0]>0]
selection={'primary_scores':scores,'chosen':None,'public_scores_used':False}
if eligible:
 chosen=max(eligible,key=lambda v:scores[v['name']]['candidate_auc']);wide=chosen['profile']=='wide';trees=int(json.loads((folder(chosen,'primary',3000 if wide else None,wide)/'fold0/result.json').read_text())['selected_trees']) if wide else 639;selection.update(chosen=chosen,trees=trees,frozen_utc=datetime.now(timezone.utc).isoformat());(ROOT/'research_stage8_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 run('research_selected_contrast_confirm',command(chosen,'confirm',trees if wide else None));c=json.loads((folder(chosen,'confirm',trees if wide else None)/'result.json').read_text())['vs_original_lgb'];selection['confirmation']=c
 if c['conditional_95pct_ci'][0]>0:
  run('research_selected_contrast_oof',command(chosen,'oof',trees if wide else None));selection['oof_result_path']=str(folder(chosen,'oof',trees if wide else None)/'result.json')
(ROOT/'research_stage8_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete')

"""Bounded original-neighbor screening, then pending CatBoost confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc']);state=ROOT/'research_stage5_queue_state.json'
def run(name,args):
 state.write_text(json.dumps({'stage':name,'status':'running','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:state.write_text(json.dumps({'stage':name,'status':'failed','error':repr(e)},indent=2)+'\n');raise
 print('DONE',name,flush=True)
run('research_neighbor_primary',['research_variant.py','--kind','lgb','--external','both_neighbors','--split','primary'])
r=json.loads((ROOT/'research_artifacts/lgb_stats_orig_both_neighbors_primary/result.json').read_text())
if r['vs_original_lgb']['delta_auc']>=.0001:
 run('research_neighbor_confirm',['research_variant.py','--kind','lgb','--external','both_neighbors','--split','confirm'])
 r=json.loads((ROOT/'research_artifacts/lgb_stats_orig_both_neighbors_confirm/result.json').read_text())
 if r['vs_original_lgb']['conditional_95pct_ci'][0]>0:
  run('research_neighbor_oof',['research_variant.py','--kind','lgb','--external','both_neighbors','--split','oof'])
run('research_cb_original_confirm',['research_variant.py','--kind','cb','--external','both','--split','confirm'])
state.write_text(json.dumps({'status':'complete','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

"""Promoted original-feature export followed by two bounded heterogeneous fits."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc'])
state=ROOT/'research_stage4_queue_state.json'
def run(name,args):
 state.write_text(json.dumps({'stage':name,'status':'running','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:state.write_text(json.dumps({'stage':name,'status':'failed','error':repr(e)},indent=2)+'\n');raise
 print('DONE',name,flush=True)
assert json.loads((ROOT/'research_stage3_queue_state.json').read_text())['status']=='complete'
assert json.loads((ROOT/'research_artifacts/lgb_stats_orig_both_oof/result.json').read_text())['stable_positive_vs_stats']
run('research_full_original',['research_full.py','--name','stats_original639'])
run('research_cb_original_primary',['research_variant.py','--kind','cb','--external','both','--split','primary'])
run('research_xgb_original_primary',['research_variant.py','--kind','xgb','--external','both','--split','primary','--trees','2000','--early-stop'])
state.write_text(json.dumps({'status':'complete','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

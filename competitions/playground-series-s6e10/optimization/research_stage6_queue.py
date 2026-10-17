"""Wait for current confirmation, then two prespecified bounded alternatives."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc']);state=ROOT/'research_stage6_queue_state.json'
state.write_text(json.dumps({'status':'waiting','stage':'stage5'},indent=2)+'\n')
while True:
 assert datetime.now(timezone.utc)<deadline
 previous=json.loads((ROOT/'research_stage5_queue_state.json').read_text())
 if previous['status']=='complete':break
 if previous['status']=='failed':raise RuntimeError(previous)
 time.sleep(5)
def run(name,args):
 state.write_text(json.dumps({'stage':name,'status':'running','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:state.write_text(json.dumps({'stage':name,'status':'failed','error':repr(e)},indent=2)+'\n');raise
 print('DONE',name,flush=True)
run('research_neural_v2_primary',['research_neural.py','--variant','v2','--epochs','80'])
run('research_residual_primary',['research_residual.py','--trees','2200','--early-stop'])
state.write_text(json.dumps({'status':'complete','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

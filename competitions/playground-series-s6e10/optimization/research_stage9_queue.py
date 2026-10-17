"""One nested stacking screen and only the selected algorithm's confirmation/OOF."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
state=ROOT/'research_stage9_queue_state.json';deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc'])
def save(stage,status,**extra):state.write_text(json.dumps({'stage':stage,'status':status,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def run(split,method):
 save('nested_stack_'+split,'running');print('START',split,method,flush=True)
 try:
  with (ROOT/f'research_nested_stack_{split}.log').open('w') as f:subprocess.run([sys.executable,'-u','research_nested_stack.py','--split',split,'--method',method],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:save('nested_stack_'+split,'failed',error=repr(e));raise
 print('DONE',split,flush=True)
run('primary','both')
r=json.loads((ROOT/'research_artifacts/nested_stack_primary/result.json').read_text())['methods'];eligible=[name for name,v in r.items() if v['comparison']['delta_auc']>=.0001 and v['comparison']['conditional_95pct_ci'][0]>0]
selection={'primary_results':r,'chosen':None,'public_scores_used':False}
if eligible:
 chosen=max(eligible,key=lambda n:r[n]['comparison']['candidate_auc']);selection.update(chosen=chosen,frozen_utc=datetime.now(timezone.utc).isoformat());(ROOT/'research_nested_stack_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 run('confirm',chosen);c=json.loads((ROOT/'research_artifacts/nested_stack_confirm/result.json').read_text())['methods'][chosen];selection['confirmation']=c
 if c['comparison']['conditional_95pct_ci'][0]>0:
  run('oof',chosen);selection['oof']=json.loads((ROOT/'research_artifacts/nested_stack_oof/result.json').read_text())['methods'][chosen]
(ROOT/'research_nested_stack_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('complete','complete')

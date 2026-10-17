"""Frozen heterogeneous confirmations/OOF, followed by a prespecified blend report."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc']);state=ROOT/'research_stage7_queue_state.json'
def run(name,args):
 state.write_text(json.dumps({'stage':name,'status':'running','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print('START',name,flush=True)
 try:
  with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 except BaseException as e:state.write_text(json.dumps({'stage':name,'status':'failed','error':repr(e)},indent=2)+'\n');raise
 print('DONE',name,flush=True)
def positive_confirmation(path,w):
 import numpy as np
 from compare_validation import compare
 a=np.load(ROOT/path/'validation.npz');b=np.load(ROOT/'research_artifacts/e1_original_both_confirm/validation.npz');assert np.array_equal(a['idx'],b['idx']);c=compare(a['target'],b['pred'],(1-w)*b['pred']+w*a['pred']);(ROOT/path/'frozen_blend_confirmation.json').write_text(json.dumps({'weight':w,'comparison':c},indent=2)+'\n');return c['conditional_95pct_ci'][0]>0
assert json.loads((ROOT/'research_stage6_queue_state.json').read_text())['status']=='complete'
run('research_neural_v2_confirm',['research_neural.py','--variant','v2','--split','confirm','--epochs','9','--fixed'])
if positive_confirmation(Path('research_artifacts/neural_v2_confirm_e9'),.25):run('research_neural_v2_oof',['research_neural.py','--variant','v2','--split','oof','--epochs','9','--fixed'])
run('research_residual_confirm',['research_residual.py','--split','confirm','--trees','581'])
if positive_confirmation(Path('research_artifacts/residual_prior05_confirm_t581'),.5):run('research_residual_oof',['research_residual.py','--split','oof','--trees','581'])
assert json.loads((ROOT/'research_cb_confirm_blend.json').read_text())['proceed_to_oof']
run('research_cb_original_oof',['research_variant.py','--kind','cb','--external','both','--split','oof'])
run('research_heterogeneous_oof_report',['research_heterogeneous_report.py'])
state.write_text(json.dumps({'status':'complete','utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

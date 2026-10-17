"""Independent confirmation/OOF of original features; no leaderboard feedback."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
deadline=datetime.fromisoformat(json.loads((ROOT/'research_protocol.json').read_text())['training_hard_cutoff_utc'])
def run(name,args):
 state={'stage':name,'status':'running','utc':datetime.now(timezone.utc).isoformat()};(ROOT/'research_stage3_queue_state.json').write_text(json.dumps(state,indent=2)+'\n')
 with (ROOT/f'{name}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=(deadline-datetime.now(timezone.utc)).total_seconds())
 print('DONE',name,flush=True)
run('research_original_both_confirm',['research_external_ablation.py','--mode','both','--split','confirm'])
r=json.loads((ROOT/'research_artifacts/e1_original_both_confirm/result.json').read_text());c=r['vs_stats']
if c['delta_auc']>0 and c['conditional_95pct_ci'][0]>0:
 run('research_original_both_oof',['research_variant.py','--kind','lgb','--external','both','--split','oof'])
(ROOT/'research_stage3_queue_state.json').write_text(json.dumps({'status':'complete','confirmation_comparison':c,'utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

"""Continue sequentially after all existing OOF fits, bounded by the frozen deadline."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,subprocess,sys
ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'research_protocol.json').read_text())
deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
state=ROOT/'research_stage2_queue_state.json'
def save(stage,**kw):state.write_text(json.dumps({'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**kw},indent=2)+'\n')
def run(stage,args):
 remaining=(deadline-datetime.now(timezone.utc)).total_seconds();assert remaining>0
 save(stage,status='running');print('START',stage,datetime.now(timezone.utc).isoformat(),flush=True)
 with (ROOT/f'{stage}.log').open('w') as f:subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 print('DONE',stage,datetime.now(timezone.utc).isoformat(),flush=True)
save('waiting_for_oof',status='waiting')
while not (ROOT/'research_artifacts/oof_v1/cb1308_oof.npz').exists():
 assert datetime.now(timezone.utc)<deadline
 time.sleep(5)
for n in ['stats639','base639','bag639','cb1308']:
 for k in range(5):assert json.loads((ROOT/f'research_artifacts/oof_v1/{n}_fold{k}/result.json').read_text())['status']=='complete'
try:
 # The initial in-memory runner used an incorrect summary dictionary key. The
 # corrected runner only verifies/reports completed fits; it skips all 20 models.
 run('research_oof_report',['research_oof.py'])
 c=json.loads((ROOT/'research_artifacts/oof_v1/comparison.json').read_text())
 if c['promote_stats']:run('research_full_stats',['research_full.py'])
 run('research_original_teacher',['research_original.py'])
 for mode in ['teacher','lookup','both']:run(f'research_external_{mode}',['research_external_ablation.py','--mode',mode])
 save('complete',status='complete')
except BaseException as e:save('failed',status='failed',error=repr(e));raise

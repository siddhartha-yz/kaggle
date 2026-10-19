"""Serial bounded progression for the frozen noise-factor model; stop at first failed gate."""
from research_w3_common import *
import subprocess
def main():
 probe=json.loads((ROOT/'research_artifacts/w3_noise_factor_probe_b2048/result.json').read_text());assert probe['status']=='complete' and probe['score_evaluated'] is False;stamp=ROOT/'research_w3_noise_factor_queue_state.json';assert not stamp.exists();state={'status':'running','started_utc':now(),'queue_pid':os.getpid(),'plan_sha256':sha(ROOT/'research_w3_noise_factor_plan.json'),'completed':[]};dump(stamp,state)
 try:
  for split in ['primary','confirm','oof']:
   start_guard();args=[str(ROOT.parent/'.venv-gpu/bin/python'),'-u',str(ROOT/'research_w3_noise_factor.py'),'--split',split,'--batch-size','2048'];left=(datetime.fromisoformat(plan()['limits']['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds();proc=None
   with (ROOT/f'research_w3_noise_factor_{split}.log').open('a') as log:
    try:
     proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);state.update(stage=split,child_pid=proc.pid,active_log=f'research_w3_noise_factor_{split}.log',last_progress_utc=now());dump(stamp,state);code=proc.wait(timeout=min(900,left-2));assert code==0,f'Noise-factor stage{split} failed code{code}'
    finally:
     if proc is not None and proc.poll() is None:
      proc.terminate()
      try:proc.wait(timeout=10)
      except subprocess.TimeoutExpired:proc.kill();proc.wait()
   result=json.loads((ROOT/f'research_artifacts/w3_noise_factor_{split}_b2048/result.json').read_text());state['completed'].append({'split':split,'vs_v6':result.get('vs_v6'),'eligible':result['eligible_for_next']});state['child_pid']=None;dump(stamp,state)
   if not result['eligible_for_next']:state.update(status='complete',outcome=f'{split}_gate_rejected',completed_utc=now());dump(stamp,state);return
  state.update(status='complete',outcome='OOF_qualified_ready_for_full',completed_utc=now());dump(stamp,state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(stamp,state);raise
if __name__=='__main__':main()

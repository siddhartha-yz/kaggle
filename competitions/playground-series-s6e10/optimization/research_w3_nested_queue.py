"""Serial, disk-backed window3 queue; refuses to repeat uncertain partial work."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
state={'status':'running','started_utc':now(),'steps':[]}
def run(script,args,out,log):
 result=ROOT/'research_artifacts'/out/'result.json'
 if result.exists():
  r=read(result)
  if r['status']=='complete':return r
  if r.get('hardware_oom') and 'probe' in out:return r
  raise RuntimeError('Existing incomplete/failed work requires inspection: '+out)
 if result.parent.exists():raise RuntimeError('Existing partial output requires inspection: '+out)
 p=read(ROOT/'research_w3_plan.json');assert datetime.now(timezone.utc)<datetime.fromisoformat(p['limits']['training_start_cutoff_utc']);state.update(active=out,active_log=log,updated_utc=now());dump(ROOT/'research_w3_nested_queue_state.json',state)
 with (ROOT/log).open('x') as f:
  proc=subprocess.Popen([sys.executable,'-u',script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);state['child_pid']=proc.pid;dump(ROOT/'research_w3_nested_queue_state.json',state);rc=proc.wait()
 r=read(result);state['steps'].append({'name':out,'exit_code':rc,'status':r['status'],'ended_utc':now()});dump(ROOT/'research_w3_nested_queue_state.json',state)
 if rc and not ('probe' in out and r.get('hardware_oom')):raise RuntimeError('Child failed: '+out)
 return r
def main():
 try:
  batch=256
  selection={'eligible_for_full':False,'chosen_variant':'tabm_nested','weights':{'v6':.75,'new_model':.25},'batch_size':batch,'plan_sha256':__import__('hashlib').sha256((ROOT/'research_w3_tabm_nested_plan.json').read_bytes()).hexdigest(),'updated_utc':now()}
  for split in ['primary','confirm','oof']:
   result=run('research_w3_tabm_nested.py',['--split',split],f'w3_tabm_nested_{split}',f'research_w3_tabm_nested_{split}.log');selection[split]=result;dump(ROOT/'research_w3_nested_selection.json',selection)
   if not result['eligible_for_next']:selection['status']=split+'_rejected';dump(ROOT/'research_w3_nested_selection.json',selection);state.update(status='complete',outcome=split+'_rejected',active=None,completed_utc=now());dump(ROOT/'research_w3_nested_queue_state.json',state);return
  selection.update(status='qualified_for_full',eligible_for_full=True,updated_utc=now());dump(ROOT/'research_w3_nested_selection.json',selection);state.update(status='complete',outcome='qualified_for_full',active=None,completed_utc=now());dump(ROOT/'research_w3_nested_queue_state.json',state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(ROOT/'research_w3_nested_queue_state.json',state);raise
if __name__=='__main__':main()

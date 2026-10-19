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
 p=read(ROOT/'research_w3_plan.json');assert datetime.now(timezone.utc)<datetime.fromisoformat(p['limits']['training_start_cutoff_utc']);state.update(active=out,active_log=log,updated_utc=now());dump(ROOT/'research_w3_queue_state.json',state)
 with (ROOT/log).open('x') as f:
  proc=subprocess.Popen([sys.executable,'-u',script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);state['child_pid']=proc.pid;dump(ROOT/'research_w3_queue_state.json',state);rc=proc.wait()
 r=read(result);state['steps'].append({'name':out,'exit_code':rc,'status':r['status'],'ended_utc':now()});dump(ROOT/'research_w3_queue_state.json',state)
 if rc and not ('probe' in out and r.get('hardware_oom')):raise RuntimeError('Child failed: '+out)
 return r
def main():
 try:
  source=read(ROOT/'research_artifacts/w3_teacher_nn_primary/result.json');assert source['status']=='complete'
  probe=run('research_w3_tabm.py',['--split','probe'],'w3_tabm_probe_b256','research_w3_tabm_probe_b256.log');batch=256
  if probe.get('hardware_oom'):batch=128;probe=run('research_w3_tabm.py',['--split','probe','--batch-size','128'],'w3_tabm_probe_b128','research_w3_tabm_probe_b128.log')
  assert probe['status']=='complete' and not probe['score_evaluated'];dump(ROOT/'research_w3_tabm_hardware.json',{'batch_size':batch,'probe':probe,'frozen_utc':now(),'scores_used':False})
  tabm=run('research_w3_tabm.py',['--split','primary','--batch-size',str(batch)],f'w3_tabm_primary_b{batch}',f'research_w3_tabm_primary_b{batch}.log');primary={'source_nn':source,'tabm':tabm};eligible=[(r['vs_v6']['candidate_auc'],v) for v,r in primary.items() if r['eligible_for_next']];selection={'primary':primary,'eligible_for_full':False,'weights':{'v6':.75,'new_model':.25},'plan_sha256':__import__('hashlib').sha256((ROOT/'research_w3_plan.json').read_bytes()).hexdigest(),'updated_utc':now()}
  if not eligible:selection['status']='both_primary_rejected';dump(ROOT/'research_w3_selection.json',selection);state.update(status='complete',outcome='no_qualified_primary',active=None,completed_utc=now());dump(ROOT/'research_w3_queue_state.json',state);return
  variant=max(eligible)[1];selection['chosen_variant']=variant;dump(ROOT/'research_w3_selection.json',selection)
  script='research_w3_teacher.py' if variant=='source_nn' else 'research_w3_tabm.py';prefix='w3_teacher_nn' if variant=='source_nn' else 'w3_tabm';extra=[] if variant=='source_nn' else ['--batch-size',str(batch)];suffix='' if variant=='source_nn' else f'_b{batch}'
  for split in ['confirm','oof']:
   result=run(script,['--split',split,*extra],f'{prefix}_{split}{suffix}',f'research_{prefix}_{split}{suffix}.log');selection[split]=result;dump(ROOT/'research_w3_selection.json',selection)
   if not result['eligible_for_next']:selection['status']=split+'_rejected';dump(ROOT/'research_w3_selection.json',selection);state.update(status='complete',outcome=split+'_rejected',active=None,completed_utc=now());dump(ROOT/'research_w3_queue_state.json',state);return
  selection.update(status='qualified_for_full',eligible_for_full=True,tabm_batch_size=batch,updated_utc=now());dump(ROOT/'research_w3_selection.json',selection);state.update(status='complete',outcome='qualified_for_full',active=None,completed_utc=now());dump(ROOT/'research_w3_queue_state.json',state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(ROOT/'research_w3_queue_state.json',state);raise
if __name__=='__main__':main()

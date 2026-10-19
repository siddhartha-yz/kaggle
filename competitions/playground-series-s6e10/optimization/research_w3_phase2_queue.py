"""Serial phase2: finish frozen primary tests, then predeclared primary-rank confirmation/OOF."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,json,numpy as np
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
state={'status':'running','started_utc':now(),'steps':[]}
def run(python,script,args,out,log):
 result=ROOT/'research_artifacts'/out/'result.json'
 if result.exists():
  r=read(result)
  if r['status']=='complete':return r
  raise RuntimeError('Existingpartial/failedworkrequiresinspection:'+out)
 if result.parent.exists():raise RuntimeError('Existingpartialdirectory:'+out)
 assert datetime.now(timezone.utc)<datetime.fromisoformat(read(ROOT/'research_w3_plan.json')['limits']['training_start_cutoff_utc']);state.update(active=out,active_log=log,updated_utc=now());dump(ROOT/'research_w3_phase2_state.json',state)
 with (ROOT/log).open('x') as f:
  proc=subprocess.Popen([python,'-u',script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);state['child_pid']=proc.pid;dump(ROOT/'research_w3_phase2_state.json',state);rc=proc.wait()
 if not result.exists():raise RuntimeError(f'Childstoppedbeforeoutput:{out},rc{rc},inspect{log}')
 r=read(result);state['steps'].append({'name':out,'exit_code':rc,'status':r['status'],'ended_utc':now()});dump(ROOT/'research_w3_phase2_state.json',state)
 if rc:raise RuntimeError('Childfailed:'+out)
 return r
def matched(name,controlpath):
 p=ROOT/'research_artifacts'/name;z=np.load(p/'validation.npz');v=np.load(ROOT/'research_artifacts/gpu_value_raw_primary_n2/validation.npz');o=np.load(ROOT/controlpath);assert all(np.array_equal(z[k],v[k]) and np.array_equal(z[k],o[k]) for k in ['idx','id','target']);c=compare(z['target'],.75*v['candidate']+.25*o['pred'],z['candidate']);r=read(p/'result.json');r['vs_matched_control']=c;r['eligible_for_next']=bool(r['eligible_for_next'] and c['delta_auc']>0);dump(p/'result.json',r);return r
def main():
 try:
  gpu=str(ROOT.parent/'.venv-gpu/bin/python');cpu=str(ROOT.parent/'.venv/bin/python');probe=read(ROOT/'research_artifacts/w3_rank_deepfm_probe_b2048/result.json');assert probe['status']=='complete' and not probe['score_evaluated'];rank=run(gpu,'research_w3_rank_deepfm.py',['--split','primary'],'w3_rank_deepfm_primary_b2048','research_w3_rank_deepfm_primary_b2048.log');rank=matched('w3_rank_deepfm_primary_b2048','research_artifacts/w3_deepfm_primary_b2048/validation.npz');cb=run(cpu,'research_w3_cb_values.py',['--split','primary'],'w3_cb_values_primary','research_w3_cb_values_primary.log');cb=matched('w3_cb_values_primary','research_artifacts/cb_stats_orig_both_primary/validation.npz');primary={'rank_deepfm':rank,'cb_values':cb};order=[v for _,v in sorted([(r['vs_v6']['candidate_auc'],v) for v,r in primary.items() if r['eligible_for_next']],reverse=True)];selection={'status':'validating_frozen_primary_order','primary':primary,'primary_eligible_order':order,'qualified_for_full':[],'results':{},'weights':{'v6':.75,'new_model':.25},'updated_utc':now()};dump(ROOT/'research_w3_phase2_selection.json',selection)
  for variant in order:
   script='research_w3_rank_deepfm.py' if variant=='rank_deepfm' else 'research_w3_cb_values.py';python=gpu if variant=='rank_deepfm' else cpu;prefix='w3_rank_deepfm' if variant=='rank_deepfm' else 'w3_cb_values';suffix='_b2048' if variant=='rank_deepfm' else '';results={};selection['results'][variant]=results
   for split in ['confirm','oof']:
    result=run(python,script,['--split',split],f'{prefix}_{split}{suffix}',f'research_{prefix}_{split}{suffix}.log');results[split]=result;dump(ROOT/'research_w3_phase2_selection.json',selection)
    if not result['eligible_for_next']:break
   if results.get('oof',{}).get('eligible_for_next'):selection['qualified_for_full'].append(variant)
  selection.update(status='complete',completed_utc=now());dump(ROOT/'research_w3_phase2_selection.json',selection);state.update(status='complete',active=None,outcome='qualified_for_full' if selection['qualified_for_full'] else 'no_qualified_phase2_route',completed_utc=now());dump(ROOT/'research_w3_phase2_state.json',state)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(ROOT/'research_w3_phase2_state.json',state);raise
if __name__=='__main__':main()

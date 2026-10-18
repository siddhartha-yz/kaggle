"""Serial, checkpointed confirmation/OOF for the single frozen family average."""
from research_w3_common import *
import argparse,subprocess,time
from research_w3_family_components import NAMES,valid_file
ORDER=['tabm_nested','tabm','deepfm','rank_deepfm','prior_deepfm','ftt']
GROUPS=[['tabm','tabm_nested'],['deepfm','rank_deepfm','prior_deepfm'],['ftt']]
def run_child(args,log,state,stamp):
 remaining=(datetime.fromisoformat(plan()['limits']['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds();assert remaining>60;proc=None
 with open(log,'a') as f:
  try:
   proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);state.update(child_pid=proc.pid,last_launch_utc=now(),current_command=args);dump(stamp,state);code=proc.wait(timeout=remaining-2);assert code==0,f'Component process failed code={code}; see{log}'
  finally:
   if proc is not None and proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['confirm','oof'],required=True);a=ap.parse_args();previous=['primary'] if a.phase=='confirm' else ['primary','confirm']
 for phase in previous:assert json.loads((ROOT/f'research_artifacts/w3_family_fusion_{phase}/result.json').read_text())['eligible_for_next']
 stamp=ROOT/f'research_w3_family_{a.phase}_queue_state.json';state={'status':'running','started_utc':now(),'phase':a.phase,'family_plan_sha256':sha(ROOT/'research_w3_family_fusion_plan.json'),'completed':[]};dump(stamp,state);df,_,y,splits,ref,_,fold=load_split(a.phase);out=ROOT/f'research_artifacts/w3_family_fusion_{a.phase}';out.mkdir(exist_ok=True);comparisons=[]
 try:
  for k,(_,vi) in enumerate(splits):
   for component in ORDER:
    path=ROOT/'research_artifacts'/NAMES[component].format(split=a.phase)/f'fold{k}/validation.npz'
    if path.exists():valid_file(path,vi,df,y);state['completed'].append({'component':component,'fold':k,'reused':True});dump(stamp,state);continue
    start_guard();env='.venv-ftt-w3' if component=='ftt' else '.venv-gpu';args=[str(ROOT.parent/env/'bin/python'),'-u',str(ROOT/'research_w3_family_components.py'),'--component',component,'--split',a.phase,'--fold',str(k)];state.update(current_component=component,current_fold=k,last_progress_utc=now());dump(stamp,state);run_child(args,ROOT/f'research_w3_family_{a.phase}_{component}_fold{k}.log',state,stamp);valid_file(path,vi,df,y);state['completed'].append({'component':component,'fold':k,'reused':False,'completed_utc':now()});dump(stamp,state)
   if a.phase=='oof':
    family=[]
    for group in GROUPS:family.append(np.mean([valid_file(ROOT/'research_artifacts'/NAMES[c].format(split='oof')/f'fold{k}/validation.npz',vi,df,y) for c in group],axis=0))
    candidate=.5*ref[vi]+.5*np.mean(family,axis=0);c=compare(y[vi],ref[vi],candidate);comparisons.append(c);np.savez_compressed(out/f'fold{k}.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],v6=ref[vi],candidate=candidate);dump(out/f'fold{k}.json',{'status':'complete','vs_v6':c,'completed_utc':now()});state['fold_comparisons']=comparisons;dump(stamp,state)
    if c['delta_auc']<=0:
     result={'status':'complete','eligible_for_next':False,'reason':'nonpositive aggregate family fold','fold_comparisons':comparisons,'completed_utc':now()};dump(out/'result.json',result);state.update(status='complete',outcome='aggregate_fold_rejected',completed_utc=now());dump(stamp,state);return
   print(a.phase,'fold',k,'all components complete',flush=True)
  run_child([str(ROOT.parent/'.venv-gpu/bin/python'),'-u',str(ROOT/'research_w3_family_fusion.py'),'--split',a.phase],ROOT/f'research_w3_family_fusion_{a.phase}.log',state,stamp);result=json.loads((out/'result.json').read_text());state.update(status='complete',eligible_for_next=result['eligible_for_next'],completed_utc=now(),current_component=None,child_pid=None);dump(stamp,state);print(json.dumps(state,indent=2),flush=True)
 except BaseException as e:state.update(status='failed',error=repr(e),ended_utc=now());dump(stamp,state);raise
if __name__=='__main__':main()

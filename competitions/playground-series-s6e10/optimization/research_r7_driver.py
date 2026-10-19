from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,os
R=Path(__file__).resolve().parent;PY=R.parent/'.venv-gpu/bin/python'
def state(status,**kw):(R/'research_r7_driver_state.json').write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'status':status,'driver_pid':os.getpid(),'whole_top100_task_completed':False,'formal_used':1,'remaining':1,**kw},indent=2)+'\n')
def call(args,log):
 with open(R/log,'a') as f:p=subprocess.run([str(PY)]+args,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,f'{args} exited{p.returncode}'
def main():
 state('resource_probe');call(['research_r7_worker.py','--probe'],'research_r7_probe.log');r=json.loads((R/'research_artifacts/r7_response_probe/result.json').read_text());assert r['status']=='complete_no_quality_score' and r['gpu_process_peak_bytes']<=1073741824
 for stage,args in [('fivefold',[]),('threefold',['--threefold'])]:
  state('training',stage=stage);call(['research_r7_worker.py']+args,'research_r7_'+stage+'.log');call(['research_r7_check.py',stage],'research_r7_'+stage+'_check.log');r=json.loads((R/'research_artifacts'/('r7_response_'+stage)/'result.json').read_text())
  if not r['eligible_for_new_partition_replication']:state('not_qualified',stage=stage,active_training=False);return
 state('qualified_awaiting_final_inference_audit',active_training=False)
if __name__=='__main__':
 try:main()
 except BaseException as e:state('failed',error=repr(e));raise

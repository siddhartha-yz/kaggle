"""Serial bounded CPU ranking fits; original full-five then fresh-three qualification."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),CPU_threads=6,GPU_used=False,parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r67_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r66_driver_state.json').read_text())['status']=='not_qualified'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            state('training',stage=stage,fold=k,active_training=True)
            with open(ROOT/f'research_r67_{stage}_fold{k}.log','a') as log:subprocess.run([sys.executable,'-u','research_r67_worker.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r67_{stage}_check.log','w') as log:subprocess.run([sys.executable,'-u','research_r67_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r67_pairlogit84_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',active_training=False,error=repr(e));raise

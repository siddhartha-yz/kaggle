"""Fixed nested-duration algorithm: fivefold, then only eligible threefold."""
import os,sys,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),
           formal_used=1,remaining=1,whole_top100_task_completed=False,**extra)
    (ROOT/'research_r10_driver_state.json').write_text(json.dumps(r,indent=2)+'\n')
def main():
    for stage,args in [('fivefold',[]),('threefold',['--threefold'])]:
        state('training',stage=stage,active_training=True)
        with open(ROOT/f'research_r10_{stage}.log','w') as log:
            subprocess.run([sys.executable,str(ROOT/'research_r10_worker.py'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((ROOT/f'research_artifacts/r10_nested_epochs_{stage}/result.json').read_text())
        if not result['eligible']:
            state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('failed',error=repr(error),active_training=False);raise

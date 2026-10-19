"""Complete all fifteen fixed aggregate-replication fits before one threefold score."""
from pathlib import Path
import subprocess,json,sys,os
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,**extra);(ROOT/'research_r64_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_artifacts/r64_viewbag_fivefold/result.json').read_text())['eligible']
    p=json.loads((ROOT/'research_r64_plan.json').read_text())
    for m in p['members']:
        view=m['name']
        if view=='r23':continue
        for k in range(3):
            state('training',view=view,fold=k,stage='threefold',active_training=True)
            with open(ROOT/f'research_r64_{view}_threefold_fold{k}.log','w') as log:subprocess.run([sys.executable,'-u','research_r64_worker.py','lean','--one-fold',str(k),'--threefold','--view',view],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(CPU),'-u','research_r64_assemble.py','--threefold','--view',view],cwd=ROOT,check=True)
    state('checking',stage='threefold',active_training=False)
    with open(ROOT/'research_r64_threefold_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r64_check.py','threefold'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((ROOT/'research_artifacts/r64_viewbag_threefold/result.json').read_text());state('qualified_for_audit' if result['eligible'] else 'not_qualified',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

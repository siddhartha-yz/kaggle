"""OnefreshGPUprocess per CB84 nested fit; onlyfull5fold gates admit3fold."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),remaining_daily_quota=6,parent_UI_quota_verified=True,actual_top500_verified=False,**extra)
    (ROOT/'research_r31_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r30_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_artifacts/r31_cb84_hardware/result.json').read_text())['status']=='passed_no_quality_score'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            state('training',stage=stage,fold=k,active_training=True)
            with open(ROOT/f'research_r31_{stage}_fold{k}.log','w') as log:subprocess.run([sys.executable,'-u','research_r31_worker.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('assembling',stage=stage,active_training=False)
        subprocess.run([str(CPU),'-u','research_r31_assemble.py',stage],cwd=ROOT,check=True)
        with open(ROOT/f'research_r31_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r31_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((ROOT/f'research_artifacts/r31_cb84_{stage}/result.json').read_text())
        if not result['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

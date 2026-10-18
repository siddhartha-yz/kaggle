"""Serial freshprocesses for4contexts perouterfold; gates before confirmation."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),remaining_daily_quota=6,parent_UI_quota_verified=True,actual_top500_verified=False,**extra)
    (ROOT/'research_r36_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r32_driver_state.json').read_text())['status']=='not_qualified'
    assert not json.loads((ROOT/'research_r33_result.json').read_text())['eligible']
    assert json.loads((ROOT/'research_artifacts/r36_tabicl48_probe_32768/result.json').read_text())['status']=='passed_no_quality_score'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for j in range(4):
                state('in_context_prediction',stage=stage,fold=k,context=j,active_work=True,active_training=False)
                with open(ROOT/f'research_r36_{stage}_fold{k}_context{j}.log','w') as log:subprocess.run([sys.executable,'-u','research_r36_worker.py',stage,str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('assembling',stage=stage,active_work=True,active_training=False)
        with open(ROOT/f'research_r36_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r36_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((ROOT/f'research_artifacts/r36_tabicl48_{stage}/result.json').read_text())
        if not result['eligible']:state('not_qualified',stage=stage,active_work=False,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_work=False,active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_work=False,active_training=False);raise

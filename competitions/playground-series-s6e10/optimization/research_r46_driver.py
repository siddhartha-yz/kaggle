"""One new auxiliary conditioning set, serial GPU auxiliary then gated CPU students."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    row=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),remaining_daily_UI_quota_observed=6,parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r46_driver_state.json').write_text(json.dumps(row,indent=2)+'\n');print('STATE',row,flush=True)
def main():
    assert json.loads((ROOT/'research_r45_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_r46_probe.json').read_text())['status']=='passed_no_SAT_read_or_quality_score'
    state('auxiliary_training',active_training=True,official_SAT_used=False)
    with open(ROOT/'research_r46_auxiliary.log','w') as log:subprocess.run([sys.executable,'-u','research_r46_auxiliary.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage in ['fivefold','threefold']:
        state('training',stage=stage,active_training=True)
        with open(ROOT/f'research_r46_{stage}.log','w') as log:subprocess.run([str(CPU),'-u','research_r46_worker.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r46_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r46_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r46_cohort_lgb111_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

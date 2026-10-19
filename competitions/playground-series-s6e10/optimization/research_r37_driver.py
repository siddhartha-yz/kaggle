"""Serialtargetfreecompression then5foldCPU6, gated3partition. No upload."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),remaining_daily_UI_quota_observed=6,parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r37_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r36_driver_state.json').read_text())['status']=='not_qualified'
    state('targetfree_rank_generation',active_training=False)
    with open(ROOT/'research_r37_generate.log','w') as log:subprocess.run([sys.executable,'-u','research_r37_generate.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage in ['fivefold','threefold']:
        state('training',stage=stage,active_training=True)
        with open(ROOT/f'research_r37_{stage}.log','w') as log:subprocess.run([sys.executable,'-u','research_r37_worker.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r37_{stage}_check.log','w') as log:subprocess.run([sys.executable,'-u','research_r37_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r37_rank_lgb_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

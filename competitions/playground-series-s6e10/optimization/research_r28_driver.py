"""Cached5foldformula -> CPU gates -> fresh3fold XGBfits onlyifqualified."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),remaining_daily_quota=None,additional_qualified_submissions_authorized=True,actual_top500_verified=False,**extra)
    (ROOT/'research_r28_driver_state.json').write_text(json.dumps(r,indent=2)+'\n')
def main():
    assert json.loads((ROOT/'research_r27_driver_state.json').read_text())['status']=='not_qualified'
    state('cached_fivefold_formula',active_training=False)
    subprocess.run([sys.executable,'-u','research_r28_bag.py'],cwd=ROOT,check=True)
    for stage in ['fivefold','threefold']:
        if stage=='threefold':
            state('training',stage=stage,active_training=True)
            with open(ROOT/'research_r28_threefold.log','w') as log:subprocess.run([sys.executable,'-u','research_r28_xgb.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with open(ROOT/f'research_r28_{stage}_check.log','w') as log:subprocess.run([sys.executable,'-u','research_r28_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads((ROOT/f'research_artifacts/r28_bag_{stage}/result.json').read_text())
        if not r['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

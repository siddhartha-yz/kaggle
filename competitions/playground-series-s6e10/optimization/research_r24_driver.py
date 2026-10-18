"""CPU6 entropy reconstruction -> fixed5fold -> conditional3fold, no uploads."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),remaining_daily_quota=None,additional_qualified_submissions_authorized=True,actual_top500_verified=False,**extra)
    (ROOT/'research_r24_driver_state.json').write_text(json.dumps(r,indent=2)+'\n')
def main():
    official=json.loads((ROOT/'research_v11_official_result.json').read_text());assert official['status']=='Complete' and official['public_auc']==.96114
    assert json.loads((ROOT/'research_r23_driver_state.json').read_text())['status']=='qualified_for_audit'
    state('auxiliary_probability_inference',active_training=False)
    with open(ROOT/'research_r24_feature_generation.log','w') as log:subprocess.run([sys.executable,'-u','research_r24_probability_features.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage,args in [('fivefold',[]),('threefold',['--threefold'])]:
        if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r24_uncertainty_lgb_officialcb_fivefold/result.json').read_text())['eligible']
        state('training',stage=stage,active_training=True)
        with open(ROOT/f'research_r24_{stage}.log','w') as log:subprocess.run([sys.executable,'-u','research_r24_worker.py',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with open(ROOT/f'research_r24_{stage}_check.log','w') as log:subprocess.run([sys.executable,'-u','research_r24_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads((ROOT/f'research_artifacts/r24_uncertainty_lgb_officialcb_{stage}/result.json').read_text())
        if not r['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

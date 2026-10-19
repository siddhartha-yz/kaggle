"""Serial freshprocesses for4contexts perouterfold; gates before confirmation."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),current_daily_quota=None,last_daily_quota_observation=9,quota_recheck_before_upload=True,actual_top500_verified=False,**extra)
    (ROOT/'research_r72_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r69_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_r72_probe_131072.json').read_text())['status']=='passed_no_quality_score'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for j in range(4):
                state('in_context_prediction',stage=stage,fold=k,context=j,active_work=True,active_training=False)
                with open(ROOT/f'research_r72_{stage}_fold{k}_context{j}.log','w') as log:subprocess.run([sys.executable,'-u','research_r72_worker.py',stage,str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('assembling',stage=stage,active_work=True,active_training=False)
        with open(ROOT/f'research_r72_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r72_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((ROOT/f'research_artifacts/r72_largecontext_raw21_{stage}/result.json').read_text())
        if not result['eligible']:state('not_qualified',stage=stage,active_work=False,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_work=False,active_training=False)
if __name__=='__main__':
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='16'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_work=False,active_training=False);raise

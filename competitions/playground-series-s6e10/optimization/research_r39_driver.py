"""Threefixednewseedblocks;serialfreshGPUprocessforeachmodel, uniform32->128."""
import subprocess,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),remaining_daily_UI_quota_observed=6,parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r39_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    evidence=json.loads((ROOT/'research_r39_variance_result.json').read_text());assert evidence['status']=='complete_predictions_only' and not evidence['official_targets_read'] and evidence['rms_native32_mean_SE']>.001
    for stage,K,args in [('fivefold',5,[]),('threefold',3,['--threefold'])]:
        for k in range(K):
            for seed in [43,44,45]:
                state('training',stage=stage,fold=k,seed=seed,active_training=True)
                with open(ROOT/f'research_r39_{stage}_fold{k}_seed{seed}.log','w') as log:subprocess.run([sys.executable,'-u','research_r39_worker.py','lean','--one-fold',str(k),'--seed',str(seed),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('assembling',stage=stage,active_training=False);subprocess.run([str(CPU),'-u','research_r39_assemble.py',stage],cwd=ROOT,check=True)
        with open(ROOT/f'research_r39_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r39_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r39_uniform_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

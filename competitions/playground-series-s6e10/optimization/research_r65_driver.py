"""Reuse five honest inner expert caches, score all five confidence gates, then fresh threefold."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r65_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r64_driver_state.json').read_text())['status']=='not_qualified'
    for stage,K,args in [('fivefold',5,[]),('threefold',3,['--threefold'])]:
        for k in range(K):
            phases=[('gate','research_r65_gate_worker.py',[stage,str(k)],CPU)]
            if stage=='threefold':phases=[('trees','research_r65_tree_worker.py',[stage,str(k)],CPU),('NN','research_r65_nn_worker.py',['lean','--one-fold',str(k),*args],sys.executable),*phases]
            for phase,program,argv,python in phases:
                state('training',stage=stage,fold=k,phase=phase,active_training=True)
                with open(ROOT/f'research_r65_{stage}_fold{k}_{phase}.log','w') as log:subprocess.run([str(python),'-u',program,*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r65_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r65_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r65_confidence_gate_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',active_training=False,error=repr(error));raise

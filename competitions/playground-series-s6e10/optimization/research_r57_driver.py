"""CPU inner trees, fresh GPU inner NN, CPU gate; serial and unchanged5then3 gates."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r57_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r56_driver_state.json').read_text())['status']=='not_qualified'
    for stage,K,args in [('fivefold',5,[]),('threefold',3,['--threefold'])]:
        for k in range(K):
            for phase,program,argv,python in [('trees','research_r57_tree_worker.py',[stage,str(k)],CPU),('NN','research_r57_nn_worker.py',['lean','--one-fold',str(k),*args],sys.executable),('gate','research_r57_gate_worker.py',[stage,str(k)],CPU)]:
                state('training',stage=stage,fold=k,phase=phase,active_training=True)
                with open(ROOT/f'research_r57_{stage}_fold{k}_{phase}.log','w') as log:subprocess.run([str(python),'-u',program,*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r57_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r57_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r57_nested_gate_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',active_training=False,error=repr(error));raise

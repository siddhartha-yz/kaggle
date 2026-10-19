"""Serial proper nested RankNet and scalar rank gate; all-five then conditional three."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r69_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def initialize(stage,K):
    import numpy as np,pandas as pd
    from sklearn.model_selection import train_test_split
    base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');df=pd.read_csv(ROOT/'data/train.csv');assert np.array_equal(base['id'],df.id) and np.array_equal(base['target'],df.satisfaction.to_numpy('int8'));y=base['target']
    for k in range(K):
        fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);run=ROOT/f'research_artifacts/r69_rank_gate_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True);p=run/'inner_indices.npz'
        arrays=dict(outer_train=fi,outer_valid=vi,inner_train=it,inner_meta=im)
        if p.exists():
            z=np.load(p);assert all(np.array_equal(z[n],v) for n,v in arrays.items())
        else:np.savez_compressed(p,**arrays)
def main():
    for stage,K,args in [('fivefold',5,[]),('threefold',3,['--threefold'])]:
        initialize(stage,K)
        for k in range(K):
            phases=[]
            if stage=='threefold':phases=[('trees','research_r69_tree_worker.py',[stage,str(k)],CPU),('NNCE','research_r69_nn_worker.py',['lean','--one-fold',str(k),*args],sys.executable)]
            phases += [('RankNet','research_r69_ranknet_worker.py',['lean','--one-fold',str(k),*args],sys.executable),('gate','research_r69_gate_worker.py',[stage,str(k)],CPU)]
            for phase,program,argv,python in phases:
                state('training',stage=stage,fold=k,phase=phase,active_training=True)
                with open(ROOT/f'research_r69_{stage}_fold{k}_{phase}.log','w') as log:subprocess.run([str(python),'-u',program,*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r69_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r69_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r69_rank_gate_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
    os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
    try:main()
    except BaseException as error:state('failed',active_training=False,error=repr(error));raise

"""Serial fresh-process contexts; five-fold gate then fresh three-fold confirmation."""
import subprocess,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    row=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,parent_only_submitter=True,**extra);(ROOT/'research_r54_driver_state.json').write_text(json.dumps(row,indent=2)+'\n');print('STATE',row,flush=True)
def main():
    assert json.loads((ROOT/'research_r47_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_r53_gpu_probe.json').read_text())['status']=='passed_no_official_SAT_or_quality'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for m in range(4):
                state('running_context',stage=stage,fold=k,context=m,active_training=True)
                with open(ROOT/f'research_r54_{stage}_fold{k}_context{m}.log','a') as log:subprocess.run([sys.executable,'-u','research_r54_worker.py',stage,str(k),str(m)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r54_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r54_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',active_training=False,error=repr(e));raise

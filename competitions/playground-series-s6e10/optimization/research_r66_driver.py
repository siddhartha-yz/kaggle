"""Serial CPU routing then fresh-process FP32 leaf contexts; unchanged five/three gates."""
import subprocess,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,parent_only_submitter=True,**extra);(ROOT/'research_r66_driver_state.json').write_text(json.dumps(r,indent=2)+'\n');print('STATE',r,flush=True)
def main():
    assert json.loads((ROOT/'research_r65_driver_state.json').read_text())['status']=='not_qualified'
    assert json.loads((ROOT/'research_r53_gpu_probe.json').read_text())['status']=='passed_no_official_SAT_or_quality'
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            state('routing',stage=stage,fold=k,active_training=True)
            with open(ROOT/f'research_r66_{stage}_fold{k}_routing.log','a') as log:subprocess.run([str(CPU),'-u','research_r66_routing.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            manifest=json.loads((ROOT/f'research_artifacts/r66_routed_tabpfn_{stage}/fold{k}/routing_manifest.json').read_text())
            for router in manifest['routers']:
                m=router['router']
                for leaf in router['leaves']:
                    j=leaf['leaf'];state('running_context',stage=stage,fold=k,router=m,leaf=j,query_rows=leaf['query_rows'],active_training=True)
                    with open(ROOT/f'research_r66_{stage}_fold{k}_router{m}_leaf{j}.log','a') as log:subprocess.run([sys.executable,'-u','research_r66_worker.py',stage,str(k),str(m),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_pending_score',stage=stage,active_training=False)
        with open(ROOT/f'research_r66_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r66_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not json.loads((ROOT/f'research_artifacts/r66_routed_tabpfn_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',active_training=False,error=repr(e));raise

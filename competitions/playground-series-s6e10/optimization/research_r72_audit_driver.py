"""Serial independent32-context audit after both frozen validation gates; no overlapping GPU jobs."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess,hashlib
from research_r72_headroom import wait_headroom
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
GPU=ROOT.parent/'.venv-tabicl-w3/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,**extra);(ROOT/'research_r72_audit_state.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
def main():
    assert json.loads((ROOT/'research_r72_driver_state.json').read_text())['status']=='qualified_for_audit';state('independent_statistics_audit')
    with open(ROOT/'research_r72_statistics_audit.log','w') as log:subprocess.run([str(CPU),'-u','research_r72_statistics_audit.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipts=[]
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for j in range(4):
                path=ROOT/f'research_artifacts/r72_independent_audit/{stage}_fold{k}_context{j}/result.json'
                if not path.exists():wait_headroom(ROOT/'research_r72_audit_headroom.json')
                state('independent_context_reconstruction',stage=stage,fold=k,context=j,completed_contexts=len(receipts))
                with open(ROOT/f'research_r72_audit_{stage}_fold{k}_context{j}.log','w') as log:subprocess.run([str(GPU),'-u','research_r72_audit_worker.py',stage,str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                r=json.loads(path.read_text());assert r['status']=='passed';receipts.append(dict(r,receipt_sha256=sha(path)))
    stats=json.loads((ROOT/'research_r72_statistics_audit.json').read_text());assert stats['status']=='passed' and stats['verified_contexts']==32
    r=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r72_plan.json'),audit_plan_sha256=sha(ROOT/'research_r72_audit_plan.json'),statistics_audit_sha256=sha(ROOT/'research_r72_statistics_audit.json'),verified_contexts=32,verified_query_chunks=stats['verified_query_chunks'],pretrained_weights_unchanged_all=True,own_context_only_preprocessing_all=True,contexts=receipts,validation=stats['validation'],actual_top500_verified=False,parent_only_submitter=True);(ROOT/'research_r72_pre_inference_audit.json').write_text(json.dumps(r,indent=2)+'\n');state('passed',verified_contexts=32)
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise

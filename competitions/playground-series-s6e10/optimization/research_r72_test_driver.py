"""Predict all20 audited contexts serially, using only immutable saved index recipes."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,hashlib
from research_r72_headroom import wait_headroom
ROOT=Path(__file__).resolve().parent
GPU=ROOT.parent/'.venv-tabicl-w3/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,**extra);(ROOT/'research_r72_test_state.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
def main():
    final=json.loads((ROOT/'research_r72_final_inference_plan.json').read_text());assert final['model_contexts']==20 and json.loads((ROOT/'research_r72_pre_inference_audit.json').read_text())['status']=='passed';receipts=[]
    for k in range(5):
        for j in range(4):
            p=ROOT/f'research_artifacts/r72_largecontext_test/fold{k}_context{j}/result.json'
            if not p.exists():wait_headroom(ROOT/'research_r72_test_headroom.json')
            state('serial_test_prediction',fold=k,context=j,completed_contexts=len(receipts))
            with open(ROOT/f'research_r72_test_fold{k}_context{j}.log','w') as log:subprocess.run([str(GPU),'-u','research_r72_test_worker.py',str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            r=json.loads(p.read_text());assert r['status']=='complete' and r['prediction_plan_sha256']==sha(ROOT/'research_r72_final_inference_plan.json');receipts.append(dict(r,receipt_sha256=sha(p)))
    result=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),prediction_plan_sha256=sha(ROOT/'research_r72_final_inference_plan.json'),contexts=receipts,total_contexts=20,parent_only_submitter=True,actual_top500_verified=False);(ROOT/'research_r72_test_complete.json').write_text(json.dumps(result,indent=2)+'\n');state('all20_contexts_complete',completed_contexts=20)
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise

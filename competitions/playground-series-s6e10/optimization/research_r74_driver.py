"""Serial origin-calibration queries then unchanged complete-five/conditional-three gates."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),new_classifier_fits=0,actual_top500_verified=False,parent_only_submitter=True,**extra);(ROOT/'research_r74_driver_state.json').write_text(json.dumps(d,indent=2)+'\n');print(d,flush=True)
def main():
 assert json.loads((ROOT/'research_r73_result.json').read_text())['status']=='not_qualified'
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   state('own_training_origin_identification_and_prediction',stage=stage,fold=k)
   with open(ROOT/f'research_r74_{stage}_fold{k}.log','w') as log:subprocess.run([sys.executable,'-u','research_r74_worker.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  state('all_folds_complete_evaluating',stage=stage)
  with open(ROOT/f'research_r74_{stage}_check.log','w') as log:subprocess.run([str(CPU),'-u','research_r74_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  if not json.loads((ROOT/f'research_artifacts/r74_member_origins_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage);return
 state('qualified_requires_independent_audit',stage='threefold')
if __name__=='__main__':
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
 os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
 try:main()
 except BaseException as error:state('failed',error=repr(error));raise

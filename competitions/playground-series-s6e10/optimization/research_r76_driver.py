"""Durable serial Mitra-v2 fivefold then conditional threefold validation."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,sys,subprocess
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def state(status,**kw):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,parent_only_submitter=True,**kw);(ROOT/'research_r76_driver_state.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def main():
 assert json.loads((ROOT/'research_r75_result.json').read_text())['status']=='not_qualified';assert json.loads((ROOT/'research_r76_probe.json').read_text())['status']=='passed_no_quality_score';assert json.loads((ROOT/'research_r76_inference_probe.json').read_text())['status']=='passed'
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   state('own_fold_microtuning_and_inference',stage=stage,fold=k)
   with open(ROOT/f'research_r76_{stage}_fold{k}.log','a') as log:subprocess.run([sys.executable,'-B','-u','research_r76_worker.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  state('all_folds_complete_evaluating',stage=stage)
  with open(ROOT/f'research_r76_{stage}_check.log','a') as log:subprocess.run([str(CPU),'-B','-u','research_r76_check.py',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  if not json.loads((ROOT/f'research_artifacts/r76_mitra_v2_{stage}/result.json').read_text())['eligible']:state('not_qualified',stage=stage);return
 state('qualified_requires_independent_audit',stage='threefold')
if __name__=='__main__':
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
 os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
 try:main()
 except BaseException as error:state('failed',error=repr(error));raise

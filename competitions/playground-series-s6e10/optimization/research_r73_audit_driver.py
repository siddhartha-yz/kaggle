"""Fresh process independent audits; no quality selection or submission."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
ROOT=Path(__file__).resolve().parent
GPU=ROOT.parent/'.venv-gpu/bin/python';CPU=ROOT.parent/'.venv/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((ROOT/'research_r73_audit_plan.json').read_text());assert plan['main_plan_sha256']==sha(ROOT/'research_r73_plan.json')
 for n,h in plan['additional_sha256'].items():assert sha(ROOT/n)==h,n
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r73_joint_density_{stage}/result.json').read_text())['eligible']
 with open(ROOT/'research_r73_statistics_audit.log','w') as log:subprocess.run([str(CPU),'-u','research_r73_statistics_audit.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 records=[]
 for stage,K in [('fivefold',5),('threefold',3)]:
  for k in range(K):
   with open(ROOT/f'research_r73_audit_{stage}_fold{k}.log','w') as log:subprocess.run([str(GPU),'-u','research_r73_audit.py',stage,str(k)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   p=ROOT/f'research_artifacts/r73_independent_audit/{stage}_fold{k}/result.json';r=json.loads(p.read_text());assert r['status']=='passed' and r['main_plan_sha256']==sha(ROOT/'research_r73_plan.json') and r['audit_plan_sha256']==sha(ROOT/'research_r73_audit_plan.json') and len(r['models'])==4;records.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p)));print('AUDIT_FOLD_COMPLETE',stage,k,flush=True)
 result=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r73_plan.json'),audit_plan_sha256=sha(ROOT/'research_r73_audit_plan.json'),statistics_sha256=sha(ROOT/'research_r73_statistics_audit.json'),verified_folds=8,verified_native_models=32,records=records,actual_top500_verified=False);(ROOT/'research_r73_pre_inference_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('ALL_AUDITS_PASSED',flush=True)
if __name__=='__main__':
 os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
 for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
 main()

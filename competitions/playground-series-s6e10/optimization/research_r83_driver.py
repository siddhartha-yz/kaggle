import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,subprocess,shutil,signal,psutil,numpy as np
from research_r83_common import *
from research_r4_common import evidence
from research_resume_policy_20261021 import check_training

def runfold(stage,k):
 check_training();assert psutil.virtual_memory().available>=6*1024**3;assert shutil.disk_usage(ROOT).free>=2*1024**3
 cmd=[sys.executable,'research_r83_worker.py',stage,str(k)]
 p=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True);tick=time.monotonic();peak=0
 try:
  while p.poll() is None:
   check_training();proc=psutil.Process(p.pid);rss=sum(q.memory_info().rss for q in [proc]+proc.children(recursive=True) if q.is_running());peak=max(peak,rss)
   assert rss<=8*1024**3,'OwnRSS8GiB';assert psutil.virtual_memory().available>=4*1024**3,'DesktopRAMreserve4GiB';assert shutil.disk_usage(ROOT).free>=2*1024**3,'Diskreserve2GiB'
   assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=256*1024**2,'Ownoutput256MiB'
   assert time.monotonic()-tick<1200,'Perfold20minresourcecheckpoint'
   time.sleep(3)
  assert p.returncode==0,f'workerexit{p.returncode}'
 except BaseException:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
  raise
 return peak

def main():
 plan=verify();OUT.mkdir(exist_ok=True);result=dict(status='running',started_utc=now(),pid=os.getpid(),plan_sha256=sha(ROOT/'research_r83_plan.json'),stages={},actual_top500_verified=False)
 dump(ROOT/'research_r83_result.json',result)
 for stage,K in [('fivefold',5),('threefold',3)]:
  peaks=[]
  for k in range(K):peaks.append(runfold(stage,k))
  z=refs(stage);r=z[0];pred=np.full(len(r['id']),np.nan)
  for k in range(K):
   run=OUT/stage/f'fold{k}';v=np.load(run/'validation.npz');ix=np.flatnonzero(r['fold']==k);assert np.array_equal(v['idx'],ix) and np.array_equal(v['id'],r['id'][ix]) and np.array_equal(v['target'],r['target'][ix]);pred[ix]=v['pred']
  assert np.isfinite(pred).all();tree=r['tree']+.46875*(pred-r['pred']);assert ((tree>=0)&(tree<=1)).all();cand=candidate(tree,z[1]['pred'],z[2]['pred'])
  dest=OUT/stage/'oof.npz';np.savez_compressed(dest,id=r['id'],target=r['target'],fold=r['fold'],pred=pred,tree=tree,candidate=cand,reference_v12=z[2]['candidate'])
  ev=evidence(r['target'],z[2]['candidate'],cand,r['fold']);passed=ev['eligible_for_new_partition_replication']
  result['stages'][stage]=dict(completed_utc=now(),all_folds_complete=True,paired_evidence=ev,passed=passed,oof_sha256=sha(dest),peak_worker_rss_bytes=max(peaks));dump(ROOT/'research_r83_result.json',result);print('STAGE',stage,json.dumps(result['stages'][stage]),flush=True)
  if not passed:
   result.update(status='closed_negative',completed_utc=now(),reason=f'{stage}failed frozen gate; no rescue or further stage',new_formal_submissions=0,new_test_inference=False);dump(ROOT/'research_r83_result.json',result);return
 result.update(status='both_partitions_passed_pending_independent_audit',completed_utc=now(),new_test_inference=False,new_formal_submissions=0);dump(ROOT/'research_r83_result.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r83_failure.json',dict(error=repr(e),utc=now()));raise

"""Fixed lean RealMLP route: eight serial independent members, capped GPU, paired OOF."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['XDG_CACHE_HOME']=str(__import__('pathlib').Path(__file__).resolve().parent.parent/'.cache-r4-lean')
from research_w3_common import ROOT,now,dump,sha,load_split
import numpy as np,pandas as pd,torch,time,gc,joblib,cloudpickle,threading,subprocess,resource
from threadpoolctl import threadpool_limits
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_r4_lean_features import LeanFeatures
# Importing CPU-only R4 evidence sets CUDA_VISIBLE_DEVICES empty; CUDA was already initialized explicitly below instead.

def resources():
 available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;assert available>=2*1024**3
 free,_=torch.cuda.mem_get_info();assert free>=1024**3
 return {'gpu_free_bytes':free,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

def main():
 import json
 pp=json.loads((ROOT/'research_r4_plan.json').read_text());cap=pp['limits']['gpu_allocator_cap_bytes'];torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available();free,_=torch.cuda.mem_get_info();assert free>=cap+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=ROOT/'research_artifacts/r4_lean_oof';out.mkdir(exist_ok=True);df,raw,y,splits,ref,old,fold=load_split('oof');cfg=dict(json.loads((ROOT/'research_gpu_value_plan.json').read_text())['config']);cfg.update(n_ens=1,wd=.015,lr_sched='flat_anneal',p_drop_sched='invsqrtp1e-3',stop_epoch=3);cfg['random_state']=42;seeds=[42,20261018,20261019,20261020,20261021,20261022,20261023,20261024]
 frozen={'frozen_utc':now(),'config':cfg,'seeds':seeds,'member_strategy':'8 independent single-member fits serially; differs from reference batched8 RNG but preserves8 members and architecture; no validation based ensemble sizing','plan_sha256':sha(ROOT/'research_r4_plan.json'),'runner_sha256':sha(__file__),'features_sha256':sha(ROOT/'research_r4_lean_features.py'),'gpu_allocator_cap_bytes':cap,'gpu_process_ceiling_bytes':1073741824,'external_source':'alvis1111/s6e10-airline-solution independent_repro/realmlp_repro_lean.py read-only mechanism comparison'}
 if (out/'plan.json').exists():
  prev=json.loads((out/'plan.json').read_text());assert all(prev[k]==frozen[k] for k in frozen if k!='frozen_utc')
 else:dump(out/'plan.json',frozen)
 peak=[0];stop=threading.Event();violated=threading.Event()
 def monitor():
  while not stop.wait(.5):
   try:
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=4)
    for line in q.stdout.splitlines():
     pid,memory=line.split(',');
     if int(pid)==os.getpid():
      peak[0]=max(peak[0],int(memory)*1024**2)
      if peak[0]>1073741824:violated.set()
   except (ValueError,subprocess.TimeoutExpired):pass
 thread=threading.Thread(target=monitor,daemon=True);thread.start();predall=np.full(len(y),np.nan)
 try:
  for k,(fi,vi) in enumerate(splits):
   run=out/f'fold{k}';run.mkdir(exist_ok=True);tr=LeanFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib');members=[]
   for j,seed in enumerate(seeds):
    path=run/f'member{j}';path.mkdir(exist_ok=True);result=path/'result.json'
    if result.exists() and json.loads(result.read_text()).get('status')=='complete':
     rec=json.loads(result.read_text());assert rec['model_sha256']==sha(path/'model.pkl');z=np.load(path/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);members.append(z['pred']);continue
    resources();assert not violated.is_set();torch.cuda.reset_peak_memory_stats();tick=time.monotonic();rec={'status':'training','started_utc':now(),'fold':k,'member':j,'seed':seed,'config':dict(cfg,random_state=seed),'feature_width':X.shape[1],'train_rows':len(fi),'valid_rows':len(vi),'epochs':[],'validation_used_for_fit_or_epoch_selection':False};dump(result,rec);dump(ROOT/'research_r4_state.json',{'updated_utc':now(),'status':'training','route':'lean_realmlp','fold':k,'member':j,'pid':os.getpid(),'active_training':True,'plan':'research_r4_plan.json'});print('START',k,j,flush=True);model=RealMLP_TD_Classifier(**rec['config']);original=TabNNModule.on_train_epoch_end;last=[time.monotonic()]
    def epoch_hook(module):
     original(module);torch.cuda.synchronize();t=time.monotonic();assert not violated.is_set(),'GPU process1GiB exceeded';rec['epochs'].append({'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'gpu_process_peak_bytes':peak[0],**resources()});last[0]=t;dump(result,rec);print('EPOCH',k,j,rec['epochs'][-1],flush=True)
    TabNNModule.on_train_epoch_end=epoch_hook
    try:
     with threadpool_limits(6):model.fit(X,y[fi])
    finally:TabNNModule.on_train_epoch_end=original
    assert len(rec['epochs'])==3 and not model.cv_alg_interface_.model.creator.is_cv;assert all(p.device.type=='cuda' for p in model.cv_alg_interface_.model.parameters());pred=model.predict_proba(V)[:,1];assert np.isfinite(pred).all();model.to('cpu')
    with open(path/'model.pkl','wb') as f:cloudpickle.dump(model,f)
    del model;gc.collect();torch.cuda.empty_cache()
    with open(path/'model.pkl','rb') as f:loaded=cloudpickle.load(f)
    loaded.to('cuda:0');again=loaded.predict_proba(V.iloc[:2048])[:,1];diff=float(np.max(np.abs(pred[:2048]-again)));assert diff<1e-7;del loaded;gc.collect();torch.cuda.empty_cache();np.savez_compressed(path/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);members.append(pred)
    rec.update(status='complete',ended_utc=now(),elapsed_seconds=time.monotonic()-tick,model_reload_error=diff,model_sha256=sha(path/'model.pkl'),prediction_sha256=sha(path/'validation.npz'),gpu_process_peak_bytes=peak[0],**resources());dump(result,rec);print('DONE',k,j,rec['elapsed_seconds'],flush=True)
   pred=np.mean(members,axis=0);predall[vi]=pred;np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],idx=vi,pred=pred,candidate=ref[vi]+.25*(pred-old[vi]));print('FOLD_COMPLETE',k,flush=True);del X,V,tr,members;gc.collect()
  assert np.isfinite(predall).all();candidate=ref+.25*(predall-old);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=candidate,v6=ref)
  from research_r4_common import evidence
  result=evidence(y,ref,candidate,fold);result.update(status='complete',route='lean_realmlp',ended_utc=now(),gpu_process_peak_bytes=peak[0],oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',result);dump(ROOT/'research_r4_state.json',{'updated_utc':now(),'status':'route_complete','route':'lean_realmlp','active_training':False,'result':str(out/'result.json'),'eligible_for_replication':result['eligible_for_new_partition_replication']});print(json.dumps(result,indent=2),flush=True)
 finally:stop.set();thread.join(timeout=5)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r4_lean_failure.json',{'status':'failed','ended_utc':now(),'error':repr(e)});raise

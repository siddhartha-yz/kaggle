"""No-score resource, latency and context-isolation probe for public frozen TabICLv2."""
from research_w3_tabicl_model import FrozenTabICL,TabICLClassifier,config
from research_w3_common import *
import time,gc,subprocess
from threadpoolctl import threadpool_limits
from research_external_ablation import external
def own_gpu_mib():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  a=line.split(',')
  if len(a)==2 and a[0].strip()==str(os.getpid()):
   try:return int(a[1].strip())
   except ValueError:pass
 return None
def cleanup(model):
 if model is not None:
  model.model_kv_cache_=None
  if hasattr(model,'model_'):model.model_.to('cpu')
 gc.collect();torch.cuda.empty_cache()
def main():
 start_guard();hw=setup_gpu();allocated_cap=1024**3;initial=own_gpu_mib()
 if initial is not None:
  overhead=max(0,initial*1024**2-torch.cuda.memory_reserved());allocated_cap=min(allocated_cap,max(256*1024**2,1024**3-overhead-128*1024**2));torch.cuda.set_per_process_memory_fraction(allocated_cap/torch.cuda.get_device_properties(0).total_memory)
 rec={'status':'running','started_utc':now(),'hardware':hw,'process_initial_gpu_mib':initial,'reduced_allocator_cap_bytes':allocated_cap,'score_evaluated':False,'query_labels_used':False,'attempts':[]};out=ROOT/'research_artifacts/w3_tabicl_large_probe';out.mkdir(exist_ok=False);dump(out/'result.json',rec);meta=json.loads((ROOT/'research_w3_tabicl_dependencies.json').read_text());assert meta['status']=='complete' and sha(meta['model']['path'])==meta['model']['sha256'];df,raw,y,splits,_,_,_=load_split('primary');fi=splits[0][0];ext,_=external('both',df.id);q=np.clip(ext.orig_teacher_lgb.to_numpy(),1e-6,1-1e-6);data=raw.copy();data['orig_teacher_logit']=np.log(q/(1-q)).astype('float32');order=np.random.default_rng(20261018).permutation(fi);chosen=None
 with threadpool_limits(6):
  for context,cache,n_est in [(8192,True,4),(8192,'repr',4),(4096,True,4),(4096,'repr',4),(2048,True,4),(2048,'repr',4)]:
   m=None;torch.cuda.reset_peak_memory_stats();attempt={'context_rows':context,'kv_cache':cache,'n_estimators':n_est,'started_utc':now()};rec['attempts'].append(attempt);dump(out/'result.json',rec)
   try:
    cfg=config(ROOT,n_est,cache);m=FrozenTabICL(**cfg);a=order[:context];b=order[context:2*context];query=order[2*context:2*context+128];t=time.monotonic();m.fit(data.iloc[a],y[a]);torch.cuda.synchronize();attempt['first_fit_seconds']=time.monotonic()-t;assert np.array_equal(m.classes_,[0,1]);t=time.monotonic();p=m.predict_proba(data.iloc[query]);torch.cuda.synchronize();attempt['predict128_seconds']=time.monotonic()-t;assert p.shape==(128,2) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    t=time.monotonic();m.fit(data.iloc[b],y[b]);torch.cuda.synchronize();attempt['warm_fit_seconds']=time.monotonic()-t;m.predict_proba(data.iloc[query]);m.fit(data.iloc[a],y[a]);again=m.predict_proba(data.iloc[query]);diff=float(np.max(np.abs(p-again)));assert diff<1e-6,'Stalecontextcacheorrefitnondeterminism';attempt['A_B_A_context_isolation_max_abs_diff']=diff;attempt['peak_allocated_bytes']=torch.cuda.max_memory_allocated();attempt['peak_reserved_bytes']=torch.cuda.max_memory_reserved();attempt['process_gpu_mib']=own_gpu_mib();attempt['estimated_primary_seconds']=1.2*(128*attempt['warm_fit_seconds']+(139927/128+128)*attempt['predict128_seconds']);gpu_epoch_guard();assert attempt['process_gpu_mib'] is None or attempt['process_gpu_mib']<=1024,'ProcessGPUmemoryexceeds1GiB';assert attempt['estimated_primary_seconds']<=1200,'Estimatedprimarylatencyexceeds20min';attempt.update(status='complete',ended_utc=now());chosen=dict(attempt);np.savez_compressed(out/'reference_probe.npz',context_idx=a,query_idx=query,pred=p);cleanup(m);del m;gc.collect();torch.cuda.empty_cache();native=TabICLClassifier(**cfg);native.fit(data.iloc[a],y[a]);ref=native.predict_proba(data.iloc[query]);nd=float(np.max(np.abs(ref-p)));assert nd<1e-6,'Frozenweightreusechangesnativeprediction';chosen['fresh_native_vs_reuse_max_abs_diff']=nd;cleanup(native);del native;gc.collect();torch.cuda.empty_cache();break
   except BaseException as e:
    attempt.update(status='failed',error=repr(e),ended_utc=now());cleanup(m);del m;gc.collect();torch.cuda.empty_cache();dump(out/'result.json',rec)
    if not (isinstance(e,torch.OutOfMemoryError) or 'ProcessGPUmemoryexceeds1GiB' in str(e) or 'Estimatedprimarylatencyexceeds20min' in str(e)):raise
  assert chosen is not None,'Noresource-safeTabICLconfiguration';rec.update(status='complete',chosen=chosen,model_sha256=meta['model']['sha256'],completed_utc=now());dump(out/'result.json',rec);dump(ROOT/'research_w3_tabicl_large_hardware.json',rec);print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()

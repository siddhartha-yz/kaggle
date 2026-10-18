"""New FP32/4096 context resource-and-own-reload proof; reused CPU mathematical audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,time,threading,gc,resource,hashlib
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier,load_fitted_tabpfn_model
from tabpfn.constants import ModelVersion
from research_r48_features import covariates,ContextFeatures
from research_r51_cache_offload import streamed_cache
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    plan=json.loads((ROOT/'research_r53_resource_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,frame=covariates('train');chosen=np.random.default_rng(20261048).permutation(len(frame));fi=chosen[:4096];vi=chosen[8192:8704];tr=ContextFeatures().fit(frame.iloc[fi]);X=tr.transform(frame.iloc[fi]);V=tr.transform(frame.iloc[vi]);y=np.arange(len(fi))%2
    cfg=dict(plan['classifier_parameters'],model_path=str(ROOT/plan['checkpoint']),categorical_features_indices=tr.categorical_indices);assert cfg['inference_precision']=='float32';cfg['inference_precision']=torch.float32;out=ROOT/'research_artifacts/r53_tabpfn_FP32_probe';out.mkdir(exist_ok=True);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        with threadpool_limits(6):
            with streamed_cache() as counters:
                start=time.monotonic();model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);model.fit(X,y);torch.cuda.synchronize();build=time.monotonic()-start;torch.cuda.reset_peak_memory_stats();start=time.monotonic();p=model.predict_proba(V);torch.cuda.synchronize();first=time.monotonic()-start;start=time.monotonic();again=model.predict_proba(V);torch.cuda.synchronize();warm=time.monotonic()-start
                assert np.array_equal(model.classes_,[0,1]) and model.n_estimators_==1 and p.shape==(512,2) and np.isfinite(p).all();repeat=float(np.max(np.abs(p-again)));assert repeat<1e-6
                CPU_cache_bytes=0
                for cache in model.executor_.kv_caches:
                    for entry in cache.kv.values():
                        assert entry.key.device.type==entry.value.device.type=='cpu';CPU_cache_bytes+=entry.key.nbytes+entry.value.nbytes
                np.savez_compressed(out/'context.npz',X=X,y=y,fit_index=fi,query_index=vi,categorical_indices=np.asarray(tr.categorical_indices));joblib.dump(tr,out/'transformer.joblib');model.to('cpu');del model;gc.collect();torch.cuda.empty_cache();z=np.load(out/'context.npz');restored=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**cfg);restored.fit(z['X'],z['y']);replay=restored.predict_proba(V);reload_error=float(np.max(np.abs(p-replay)));assert reload_error<1e-6
                assert counters['build_column_chunks']>0 and counters['query_column_chunks']>0
                rec=dict(status='passed_no_official_SAT_or_quality',features=49,context_rows=4096,query_rows=512,n_estimators=1,official_SAT_values_parsed=False,synthetic_labels=True,quality_scores_computed=False,prior_CPU_reference_max_error=json.loads((ROOT/'research_r52_precision_diagnostic.json').read_text())['errors']['CPU_stock_vs_CPU_stream']['max'],inference_precision='float32',cross_backend_equivalence_claim=False,model_context_reload_error=reload_error,repeat_error=repeat,cache_build_seconds=build,first_query_seconds=first,warm_query_seconds=warm,CPU_KV_cache_bytes=CPU_cache_bytes,adapter_counters=dict(counters),peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,model_version='v2',checkpoint_sha256=sha(ROOT/plan['checkpoint']),plan_sha256=sha(ROOT/'research_r53_resource_plan.json'),attribution='Built with PriorLabs-TabPFN');assert 0<peak[0]<=CEILING and rec['peak_RSS_bytes']<=8589934592;dump(ROOT/'research_r53_gpu_probe.json',rec);print(json.dumps(rec),flush=True)
    except BaseException as e:dump(ROOT/'research_r53_gpu_probe.json',dict(status='failed_resource_or_equivalence_no_quality',error=repr(e),peak_gpu_process_bytes=peak[0],quality_scores_computed=False,plan_sha256=sha(ROOT/'research_r53_resource_plan.json')));raise
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

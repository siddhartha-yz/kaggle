"""Built with PriorLabs-TabPFN. V2/8192context/49fields hardware proof; synthetic SAT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4');os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import json,time,threading,gc,resource,hashlib
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier,save_fitted_tabpfn_model,load_fitted_tabpfn_model
from tabpfn.constants import ModelVersion
from research_r48_features import covariates,ContextFeatures
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    plan=json.loads((ROOT/'research_r48_resource_plan.json').read_text());assert json.loads((ROOT/'research_r48_cpu_preflight.json').read_text())['status']=='passed_synthetic_only'
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,frame=covariates('train');chosen=np.random.default_rng(20261048).permutation(len(frame));fi=chosen[:8192];vi=chosen[8192:8704];assert not np.intersect1d(fi,vi).size
    transformer=ContextFeatures().fit(frame.iloc[fi]);X=transformer.transform(frame.iloc[fi]);V=transformer.transform(frame.iloc[vi]);y=np.arange(len(fi))%2
    out=ROOT/'research_artifacts/r48_tabpfn_v2_gpu_probe';out.mkdir(exist_ok=True);joblib.dump(transformer,out/'transformer.joblib');peak=[0];stop=threading.Event();monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        config=dict(plan['classifier_parameters']);config['model_path']=str(ROOT/plan['checkpoint']);config['categorical_features_indices']=transformer.categorical_indices
        with threadpool_limits(6):
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,**config);model.fit(X,y);assert np.array_equal(model.classes_,[0,1]) and model.n_estimators_==1
            torch.cuda.reset_peak_memory_stats();start=time.monotonic();p=model.predict_proba(V);torch.cuda.synchronize();first=time.monotonic()-start;start=time.monotonic();again=model.predict_proba(V);torch.cuda.synchronize();warm=time.monotonic()-start
            assert p.shape==(512,2) and np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-6;repeat=float(np.max(np.abs(p-again)));assert repeat<1e-6
            model.to('cpu');save_fitted_tabpfn_model(model,out/'model.tabpfn_fit');del model;gc.collect();torch.cuda.empty_cache();restored=load_fitted_tabpfn_model(out/'model.tabpfn_fit',device='cuda:0');replay=restored.predict_proba(V);error=float(np.max(np.abs(p-replay)));assert error<1e-6
        record=dict(status='passed_no_official_SAT_or_quality',features=49,context_rows=8192,query_rows=512,n_estimators=1,official_SAT_values_parsed=False,synthetic_labels=True,quality_scores_computed=False,model_version='v2',checkpoint_sha256=sha(ROOT/plan['checkpoint']),first_query_seconds=first,warm_query_seconds=warm,repeat_error=repeat,model_reload_error=error,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,plan_sha256=sha(ROOT/'research_r48_resource_plan.json'),attribution='Built with PriorLabs-TabPFN');assert 0<peak[0]<=CEILING and record['peak_RSS_bytes']<=8589934592;dump(ROOT/'research_r48_gpu_probe.json',record);print(json.dumps(record),flush=True)
    except BaseException as e:dump(ROOT/'research_r48_gpu_probe.json',dict(status='failed_hardware_no_quality',error=repr(e),peak_gpu_process_bytes=peak[0],quality_scores_computed=False,plan_sha256=sha(ROOT/'research_r48_resource_plan.json')));raise
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

"""Built with PriorLabs-TabPFN. CPU-only API/reload proof on synthetic data, no CSV reads."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT.parent/'.hf-tabpfn-w4')
os.environ['SKB_DATA_DIRECTORY']=str(ROOT.parent/'.skrub-tabpfn-w4')
import time,json,hashlib,resource,threading,gc
import numpy as np,torch,cloudpickle
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from research_r40_worker import watch_cpu

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((ROOT/'research_r48_cpu_preflight_plan.json').read_text());assert plan['source_sha256']==sha(__file__)
    for n,h in plan['input_sha256'].items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);rng=np.random.default_rng(20261048);X=rng.normal(size=(128,48)).astype('float32');X[:,:4]=rng.integers(0,3,size=(128,4));y=np.arange(128)%2;V=rng.normal(size=(64,48)).astype('float32');V[:,:4]=rng.integers(0,3,size=(64,4))
    checkpoint=ROOT/plan['checkpoint'];out=ROOT/'research_artifacts/r48_tabpfn_cpu_preflight';out.mkdir(exist_ok=True);peak=[0];stop=threading.Event();monitor=threading.Thread(target=watch_cpu,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        with threadpool_limits(6):
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,model_path=checkpoint,n_estimators=1,auto_scale_n_estimators=False,device='cpu',categorical_features_indices=[0,1,2,3],random_state=42,n_preprocessing_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',memory_saving_mode=True,enable_torch_compile=False,show_progress_bar=False)
            model.fit(X,y);assert np.array_equal(model.classes_,[0,1]);p=model.predict_proba(V);again=model.predict_proba(V);assert p.shape==(64,2) and np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-6
            repeat=float(np.max(np.abs(p-again)));assert repeat<1e-7
            with (out/'model.pkl').open('wb') as f:cloudpickle.dump(model,f)
            del model;gc.collect()
            with (out/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
            error=float(np.max(np.abs(p-model.predict_proba(V))));assert error<1e-7
        record=dict(status='passed_synthetic_only',official_CSVs_loaded=False,official_SAT_values_parsed=False,quality_scores_computed=False,model_family='TabPFN-v2',checkpoint_sha256=sha(checkpoint),library_version=__import__('tabpfn').__version__,CPU_threads=torch.get_num_threads(),preprocessing_jobs=1,training_context_rows=128,query_rows=64,features=48,categorical_features_indices=[0,1,2,3],repeat_error=repeat,reload_error=error,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,plan_sha256=sha(ROOT/'research_r48_cpu_preflight_plan.json'),attribution='Built with PriorLabs-TabPFN',no_network_model_loading=True)
        (ROOT/'research_r48_cpu_preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

"""Synthetic-label8192-context fixed-schedule old/new FFN parity and cache isolation."""
from research_r36_probe import *
def main():
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    p=json.loads((ROOT/'research_r36_resource_plan.json').read_text())
    for n,h in {**p['source_sha256'],**p['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    out=ROOT/'research_artifacts/r36_memory_parity';out.mkdir(exist_ok=False)
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cols=list(pd.read_csv(ROOT/'data/train.csv',nrows=0));df=pd.read_csv(ROOT/'data/train.csv',usecols=[c for c in cols if c!='satisfaction']);X=features(df.drop(columns='id'),auxiliary(df.id));order=np.random.default_rng(20261031).permutation(len(df));A=order[:8192];B=order[8192:16384];Q=order[16384:16512];y=np.arange(8192,dtype=np.int64)%2
        cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=['none'],offload_mode='cpu',inference_config=inference_config())
        with threadpool_limits(6):
            with bounded_offload(ff_chunk_rows=4096):
                old=FrozenTabICL(**cfg);old.fit(X.iloc[A],y);reference=old.predict_proba(X.iloc[Q]);del old;gc.collect();torch.cuda.empty_cache()
            with streamed_ffn() as counters:
                m=FrozenTabICL(**cfg);m.fit(X.iloc[A],y);prediction=m.predict_proba(X.iloc[Q]);error=float(np.max(np.abs(reference-prediction)));assert error==0.0
                m.fit(X.iloc[B],y);m.predict_proba(X.iloc[Q]);m.fit(X.iloc[A],y);again=m.predict_proba(X.iloc[Q]);isolation=float(np.max(np.abs(prediction-again)));assert isolation==0.0 and counters['preallocated_ff_calls']>0
            result=dict(status='passed_no_quality_score',completed_utc=now(),official_SAT_column_loaded=False,synthetic_labels=True,context_rows=8192,feature_width=48,query_rows=128,same_schedule_old_new_max_probability_error=error,A_B_A_error=isolation,counters=dict(counters),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitation='PreservesR22fixedmicrobatchschedule;stockauto-batchFP16equivalenceisnotclaimed.')
            dump(out/'result.json',result);print(result,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

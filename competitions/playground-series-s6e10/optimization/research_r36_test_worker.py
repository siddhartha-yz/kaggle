"""One saved context reconstructed and replayed before bounded test prediction."""
from research_r36_worker import *
def main(k,j):
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    final=json.loads((ROOT/'research_r36_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r36_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r36_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r36_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_context_records']==32 and audit['reconstructed_contexts']==32 and audit['large32768_FFN_bitwise_error']==0
    src=ROOT/f'research_artifacts/r36_tabicl48_fivefold/fold{k}/context{j}';rec=json.loads((src/'result.json').read_text());context=np.load(src/'context.npz')['idx'];v=np.load(src/'validation.npz');out=ROOT/f'research_artifacts/r36_tabicl48_full/fold{k}/context{j}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_text());assert r['status']=='complete' and r['final_plan_sha256']==sha(ROOT/'research_r36_final_inference_plan.json') and r['prediction_sha256']==sha(out/'test.npz');return
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(test)==299844 and np.array_equal(test.id,sample.id)
        y=train.satisfaction.to_numpy('int8');X=features(train.drop(columns=['id','satisfaction']),auxiliary(train.id));T=features(test.drop(columns='id'),auxiliary(test.id,test=True));assert list(X)==list(T)
        cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=[rec['norm_method']],random_state=42+j,offload_mode='cpu',inference_config=inference_config());tick=time.monotonic()
        with threadpool_limits(6),streamed_ffn() as counters:
            m=FrozenTabICL(**cfg);m.fit(X.iloc[context],y[context]);assert np.array_equal(m.classes_,[0,1]);again=m.predict_proba(X.iloc[v['idx'][:512]])[:,1];replay=float(np.max(np.abs(again-v['pred'][:512])));assert replay<1e-6
            p=np.full(len(test),np.nan);resume=None
            for start in range(0,len(test),512):
                take=np.arange(start,min(start+512,len(test)));file=out/f'query{start:06d}.npz'
                if file.exists():
                    q=np.load(file);assert np.array_equal(q['id'],test.id.iloc[take]);pv=q['pred']
                    if resume is None:
                        again=m.predict_proba(T.iloc[take])[:,1];resume=float(np.max(np.abs(again-pv)));assert resume<1e-6
                else:
                    assert torch.cuda.mem_get_info()[0]>=1024**3;pv=m.predict_proba(T.iloc[take])[:,1];assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();np.savez_compressed(file,id=test.id.iloc[take].to_numpy(),pred=pv)
                p[take]=pv
                if start%(32*512)==0 or start+512>=len(test):
                    r=dict(status='test_prediction',fold=k,context=j,predicted_rows=int(take[-1])+1,updated_utc=now(),pid=os.getpid(),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0]);dump(out/'state.json',r);dump(ROOT/'research_r36_test_state.json',r);print('TEST_PROGRESS',r,flush=True)
            again=m.predict_proba(T.iloc[:512])[:,1];repeat=float(np.max(np.abs(again-p[:512])));assert repeat<1e-6 and 0<peak[0]<=1073741824
            np.savez_compressed(out/'test.npz',id=test.id.to_numpy(),pred=p);r=dict(status='complete',completed_utc=now(),fold=k,context=j,context_sha256=sha(src/'context.npz'),validation_reconstruction_error=replay,test_repeat_error=repeat,resume_replay_error=resume,prediction_sha256=sha(out/'test.npz'),rows=len(test),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,final_plan_sha256=sha(ROOT/'research_r36_final_inference_plan.json'),pretrained_weight_update=False);dump(out/'result.json',r);print('TEST_COMPLETE',r,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))

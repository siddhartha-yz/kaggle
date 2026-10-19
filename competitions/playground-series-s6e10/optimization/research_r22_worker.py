"""Frozen 16k global TabICL; outer-only context/preprocessing, no interim scoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import bounded_offload,inference_config,CAP,CEILING
import torch,numpy as np,pandas as pd,json,time,gc,threading,subprocess,resource,hashlib,psutil,sys
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def pool(a,b):
    a=np.clip(a.astype('float64'),1e-6,1-1e-6);b=np.clip(b.astype('float64'),1e-6,1-1e-6)
    q=.875*np.log(a/(1-a))+.125*np.log(b/(1-b));return 1/(1+np.exp(-q))
def main(stage):
    assert stage in ('fivefold','threefold')
    plan=json.loads((ROOT/'research_r22_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r22_global_tabicl_fivefold/result.json').read_text())['eligible']
    out=ROOT/f'research_artifacts/r22_global_tabicl_{stage}';out.mkdir(exist_ok=True)
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event();process=psutil.Process()
    def monitor():
        try:
            while not stop.wait(.25):
                q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
                for line in q.stdout.splitlines():
                    try:pid,memory=[int(x.strip()) for x in line.split(',')]
                    except ValueError:continue
                    if pid==os.getpid():peak[0]=max(peak[0],memory*2**20)
                available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;rss=process.memory_info().rss
                if peak[0]>CEILING or available<2*1024**3 or rss>8*1024**3:
                    dump(out/'resource_failure.json',dict(peak_gpu_process_bytes=peak[0],available_RAM=available,RSS=rss));os._exit(86)
        except BaseException as error:dump(out/'resource_failure.json',dict(monitor_error=repr(error)));os._exit(87)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
        z=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=z['fold'];base=z['candidate']
        assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and raw.shape==(699635,21)
        predall=np.full(len(y),np.nan);cfg=dict(config(ROOT,n_estimators=4,cache=True),offload_mode='cpu',inference_config=inference_config())
        with threadpool_limits(6):
            for k in sorted(np.unique(fold)):
                k=int(k);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k)
                context=np.random.default_rng(20261030+k+(0 if stage=='fivefold' else 10000)).permutation(fi)[:16384]
                assert len(np.unique(context))==16384 and np.isin(context,fi).all() and not np.isin(context,vi).any()
                run=out/f'fold{k}';run.mkdir(exist_ok=True);stamp=run/'result.json'
                if stamp.exists():
                    previous=json.loads(stamp.read_text());assert previous['plan_sha256']==sha(ROOT/'research_r22_plan.json') and previous['runner_sha256']==sha(__file__)
                    if previous['status']=='complete':
                        assert previous['prediction_sha256']==sha(run/'validation.npz');q=np.load(run/'validation.npz');assert np.array_equal(q['idx'],vi) and np.array_equal(q['id'],df.id.iloc[vi]);predall[vi]=q['pred'];continue
                np.savez_compressed(run/'context.npz',idx=context,id=df.id.iloc[context].to_numpy())
                rec=dict(status='in_context_prediction',stage=stage,fold=k,started_utc=now(),plan_sha256=sha(ROOT/'research_r22_plan.json'),runner_sha256=sha(__file__),context_sha256=sha(run/'context.npz'),context_rows=len(context),train_rows=len(fi),valid_rows=len(vi),validation_used_in_context_or_preprocessing=False,pretrained_weight_update=False)
                dump(stamp,rec);dump(ROOT/'research_r22_state.json',dict(rec,pid=os.getpid(),active_work=True,active_training=False,device='GPU bounded process with CPU KV cache',predicted_rows=0))
                tick=time.monotonic();torch.cuda.reset_peak_memory_stats();m=None
                try:
                    with bounded_offload(ff_chunk_rows=4096) as counters:
                        m=FrozenTabICL(**cfg);m.fit(raw.iloc[context],y[context]);assert np.array_equal(m.classes_,[0,1]);rec['fit_seconds']=time.monotonic()-tick
                        cache_devices=sorted({str(e.key.device) for c in m.model_kv_cache_.values() for cache in (c.col_cache,c.icl_cache) if cache for e in cache.kv.values() if e.is_valid()});assert cache_devices==['cpu'];rec['cache_devices']=cache_devices
                        pred=np.full(len(vi),np.nan);replay=None
                        for start in range(0,len(vi),512):
                            take=vi[start:start+512];checkpoint=run/f'query{start:06d}.npz'
                            if checkpoint.exists():
                                q=np.load(checkpoint);assert np.array_equal(q['idx'],take) and np.array_equal(q['id'],df.id.iloc[take]);pv=q['pred']
                                if replay is None:
                                    again=m.predict_proba(raw.iloc[take])[:,1];replay=float(np.max(np.abs(again-pv)));assert replay<1e-5
                            else:
                                assert torch.cuda.mem_get_info()[0]>=1024**3
                                pv=m.predict_proba(raw.iloc[take])[:,1];assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all()
                                np.savez_compressed(checkpoint,idx=take,id=df.id.iloc[take].to_numpy(),pred=pv)
                            pred[start:start+len(take)]=pv
                            if start%(16*512)==0 or start+512>=len(vi):
                                rec.update(predicted_rows=min(start+512,len(vi)),updated_utc=now(),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0]);dump(stamp,rec);dump(ROOT/'research_r22_state.json',dict(rec,pid=os.getpid(),active_work=True,active_training=False));print('PROGRESS',stage,k,rec['predicted_rows'],round(rec['seconds'],2),flush=True)
                        again=m.predict_proba(raw.iloc[vi[:512]])[:,1];repeat=float(np.max(np.abs(again-pred[:512])));assert repeat<1e-6
                        rec.update(repeat_error=repeat,resume_refit_replay_error=replay,counters=dict(counters),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    predall[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=pool(base[vi],pred))
                    rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,prediction_sha256=sha(run/'validation.npz'),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(stamp,rec);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True)
                finally:del m;gc.collect();torch.cuda.empty_cache()
        assert np.isfinite(predall).all();np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=pool(base,predall),reference_v10=base)
        dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));dump(ROOT/'research_r22_state.json',dict(status='stage_complete_pending_score',stage=stage,active_work=False,active_training=False,updated_utc=now()))
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as error:dump(ROOT/'research_r22_failure.json',dict(error=repr(error),updated_utc=now()));raise

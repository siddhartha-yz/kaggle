"""Largest-fold8tree GPU resource test; no official validation score."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r16')
Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,time,gc,threading,subprocess,signal,resource
import numpy as np,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_r5_auto_te import data
from research_external_ablation import external
from research_r16_cb_features import CatDistanceFeatures
from datetime import datetime,timezone
CEILING=1073741824
def now():return datetime.now(timezone.utc).isoformat()
def main():
    # The reused CPU-only data helper clears visibility during import.
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    fraction=.06 if '--06' in os.sys.argv else .04
    trees=64 if '--64' in os.sys.argv else 8
    out=ROOT/'research_artifacts'/(f'r16_cb_gpu_speed64_{int(fraction*100):02}' if trees==64 else f'r16_cb_gpu_probe_{int(fraction*100):02}');out.mkdir(exist_ok=True)
    rec=dict(status='running',started_utc=now(),gpu_ram_part=fraction,no_quality_score=True,
             gpu_process_ceiling_bytes=CEILING,context=f'largest559708row train fold,58features,depth10,{trees}trees')
    def dump(): (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
    dump();peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:
                        rec.update(status='GPU_ceiling_exceeded_stop',peak_gpu_process_bytes=peak[0],completed_utc=now());dump();os._exit(86)
    def timeout(signum,frame):raise TimeoutError('R16 fullresourceprobe180seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        df,raw,y,fold,*_=data('fivefold');fi=np.flatnonzero(fold!=0);vi=np.flatnonzero(fold==0)
        ext,_=external('both',df.id);tr=CatDistanceFeatures(True)
        X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True))
        V=tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True),ext.iloc[vi[:2048]].reset_index(drop=True))
        params=dict(iterations=trees,depth=10,learning_rate=.04,l2_leaf_reg=10,loss_function='Logloss',eval_metric='AUC',
            task_type='GPU',devices='0',gpu_ram_part=fraction,thread_count=6,border_count=254,max_ctr_complexity=1,
            one_hot_max_size=2,boosting_type='Plain',random_seed=20261018,allow_writing_files=False,verbose=16 if trees==64 else False)
        tick=time.monotonic()
        with threadpool_limits(6):
            model=CatBoostClassifier(**params);model.fit(X,y[fi],cat_features=tr.cat_columns)
            assert model.tree_count_==trees and np.array_equal(model.classes_,[0,1])
            p=model.predict_proba(V,thread_count=6)[:,1];assert np.isfinite(p).all()
            model.save_model(str(out/'model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(out/'model.cbm'))
            error=float(np.max(np.abs(p-loaded.predict_proba(V,thread_count=6)[:,1])));assert error<1e-12
        rec.update(status='passed_no_quality_score',completed_utc=now(),train_rows=len(fi),feature_width=X.shape[1],
            category_columns=tr.cat_columns,seconds=time.monotonic()-tick,model_reload_error=error,
            peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            actual_params=model.get_all_params())
        assert 0<peak[0]<=CEILING;dump();print(json.dumps({k:v for k,v in rec.items() if k!='actual_params'},indent=2),flush=True)
    except BaseException as error:
        rec.update(status='failed',error=repr(error),completed_utc=now(),peak_gpu_process_bytes=peak[0]);dump();raise
    finally:signal.alarm(0);stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

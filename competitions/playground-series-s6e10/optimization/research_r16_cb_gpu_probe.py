"""Tiny synthetic CatBoost CUDA compatibility test; no official rows or scores."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,time,threading,subprocess,signal,resource
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,__version__
R=Path(__file__).resolve().parent
OUT=R/'research_r16_cb_gpu_tiny.json'
CEILING=1073741824
def now():return datetime.now(timezone.utc).isoformat()
def dump(x):OUT.write_text(json.dumps(x,indent=2)+'\n')
def main():
    rec=dict(status='running',started_utc=now(),official_data_used=False,no_quality_score=True,
             catboost_version=__version__,gpu_process_ceiling_bytes=CEILING)
    dump(rec);peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:
                        dump(dict(rec,status='GPU_ceiling_exceeded_stop',peak_gpu_process_bytes=peak[0],completed_utc=now()));os._exit(86)
    def timeout(signum,frame):raise TimeoutError('TinyGPU compatibility probe120s limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        rng=np.random.default_rng(20261029);n=4096
        X=pd.DataFrame({f'x{j}':rng.normal(size=n) for j in range(6)})
        X['distance_token']=rng.integers(0,128,n).astype(str);X['class_token']=rng.integers(0,3,n).astype(str)
        y=rng.integers(0,2,n,dtype='int8')
        params=dict(iterations=1,depth=8,learning_rate=.04,l2_leaf_reg=10,loss_function='Logloss',
            task_type='GPU',devices='0',gpu_ram_part=.01,thread_count=6,border_count=254,max_ctr_complexity=1,
            one_hot_max_size=2,random_seed=20261018,allow_writing_files=False,verbose=False)
        tick=time.monotonic();model=CatBoostClassifier(**params);model.fit(X,y,cat_features=['distance_token','class_token'])
        assert model.tree_count_==1 and np.array_equal(model.classes_,[0,1])
        params_actual=model.get_all_params();assert params_actual['task_type']=='GPU'
        rec.update(status='passed_tiny_synthetic_GPU',completed_utc=now(),seconds=time.monotonic()-tick,
            peak_gpu_process_bytes=peak[0],params=params,actual_task_type=params_actual['task_type'],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert 0<peak[0]<=CEILING
        dump(rec);print(json.dumps(rec,indent=2),flush=True)
    except BaseException as error:
        rec.update(status='failed',completed_utc=now(),error=repr(error),peak_gpu_process_bytes=peak[0]);dump(rec);raise
    finally:signal.alarm(0);stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

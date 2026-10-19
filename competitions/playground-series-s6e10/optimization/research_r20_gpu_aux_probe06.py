"""Largest fivefold auxiliary multiclass GPU sizing test; no satisfaction read/scored."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r20');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,time,threading,subprocess,resource,gc,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
CEILING=1073741824
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
def main():
    plan=json.loads((ROOT/'research_r20_resource06_plan.json').read_text());assert plan['runner_sha256']==sha(__file__)
    for n,h in plan['data_sha256'].items():assert sha(ROOT/n)==h
    cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']]
    a=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*cols]);b=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*cols]);raw=pd.concat([a[cols],b[cols]],ignore_index=True)
    assert raw.shape==(999479,21) and 'satisfaction' not in raw
    target=raw['Inflight wifi service'].to_numpy('int8');X=raw.drop(columns='Inflight wifi service').copy()
    cats=X.select_dtypes(exclude='number').columns.tolist()
    for c in cats:X[c]=X[c].fillna('__MISSING__').astype(str)
    fi,vi=list(KFold(5,shuffle=True,random_state=42).split(X))[-1];assert len(fi)==799584
    out=ROOT/'research_artifacts/r20_multiclass_gpu_probe06';out.mkdir(exist_ok=True)
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(x.strip()) for x in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*2**20)
                    if peak[0]>CEILING:dump(out/'resource_failure.json',dict(peak_gpu_process_bytes=peak[0]));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();tick=time.monotonic()
    try:
        with threadpool_limits(6):
            model=CatBoostClassifier(**plan['parameters']);model.fit(X.iloc[fi],target[fi],cat_features=cats)
            assert model.get_all_params()['task_type']=='GPU' and model.tree_count_==300
            p=model.predict_proba(X.iloc[vi[:2048]],thread_count=6);assert p.shape==(2048,6) and np.isfinite(p).all()
            model.save_model(str(out/'model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(out/'model.cbm'))
            error=float(np.max(np.abs(p-loaded.predict_proba(X.iloc[vi[:2048]],thread_count=6))));assert error<1e-12
        r=dict(status='passed_no_satisfaction_read_or_quality_score',parameters=plan['parameters'],classes=model.classes_.tolist(),train_rows=len(fi),features=20,categorical_columns=cats,
            seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],gpu_process_ceiling_bytes=CEILING,reload_prediction_error=error,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,runner_sha256=sha(__file__))
        assert 0<peak[0]<=CEILING;dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

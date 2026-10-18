"""Largest-row GPU CatBoost84 probe with synthetic labels, no SAT scores."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r31');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
from research_r4_common import now,dump,sha,guard
from research_r31_features import ConditionalCatFeatures
from research_r23_loader import auxiliary
from research_r23_guard import watch
import json,time,threading,gc,resource,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0';plan=json.loads((ROOT/'research_r31_resource_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    out=ROOT/'research_artifacts/r31_cb84_hardware';out.mkdir(exist_ok=True);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']];df=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*cols]);raw=df[cols];fake=(np.arange(len(df))%2).astype('int8');assert len(df)==699635
        z=np.load(ROOT/'research_artifacts/original_teacher_v1/train_features.npz');assert np.array_equal(z['id'],df.id);ext=pd.DataFrame(z['values'],columns=z['columns']);aux=auxiliary(df.id)
        tr=ConditionalCatFeatures();X=tr.fit_transform(raw,fake,ext,aux);assert not tr.base.te_maps and X.shape==(699635,84)
        cfg=dict(plan['parameters'],iterations=8,verbose=False);guard();tick=time.monotonic()
        with threadpool_limits(6):
            model=CatBoostClassifier(**cfg);model.fit(X,fake,cat_features=tr.cat_columns);assert model.get_all_params()['task_type']=='GPU' and model.tree_count_==8
            p=model.predict_proba(X.iloc[:2048],thread_count=6);model.save_model(str(out/'probe_model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(out/'probe_model.cbm'));again=loaded.predict_proba(X.iloc[:2048],thread_count=6);err=float(np.max(np.abs(p-again)));assert err<1e-12
        assert 0<peak[0]<=1073741824
        r=dict(status='passed_no_quality_score',completed_utc=now(),official_SAT_column_loaded=False,synthetic_labels=True,rows=len(df),features=84,iterations=8,actual_task_type='GPU',category_columns=tr.cat_columns,seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,reload_error=err,runner_sha256=sha(__file__))
        dump(out/'result.json',r);print(json.dumps(r,indent=2),flush=True)
    except BaseException as error:dump(out/'result.json',dict(status='failed_no_quality_score',error_type=type(error).__name__,error=str(error),peak_gpu_process_bytes=peak[0],official_SAT_column_loaded=False));raise
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

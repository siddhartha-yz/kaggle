"""Full699635x84 global-group PairLogit resource proof with synthetic labels only."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,dump,now,sha
from research_r40_worker import watch_cpu
from research_r23_loader import auxiliary
from research_external_ablation import external
from research_r67_pairs import pairs
import json,time,threading,resource,joblib,gc,numpy as np,pandas as pd
from catboost import CatBoostRanker,Pool
from threadpoolctl import threadpool_limits

def main():
    plan=json.loads((ROOT/'research_r67_resource_plan_v2.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if str(path).endswith('data/train.csv') and kwargs.get('nrows')!=0:assert 'usecols' in kwargs and 'satisfaction' not in kwargs['usecols']
        return original(path,*args,**kwargs)
    pd.read_csv=guarded;stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r67_probe_v2_resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']];frame=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*columns]);raw=frame[columns];assert len(raw)==699635
        tr=joblib.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/transformer.joblib');assert tr.base.n==len(raw) and tr.base.te_maps==[];ext,_=external('both',frame.id);aux=auxiliary(frame.id);X=tr.transform(raw,ext,aux);assert X.shape==(699635,84);cats=X.select_dtypes('category').columns.tolist()
        for c in cats:X[c]=X[c].astype(str)
        y=(np.arange(len(X))%2).astype('int8');graph,weights=pairs(y);generated=time.monotonic();assert weights.sum()==len(X)*8
        with threadpool_limits(6):
            pool=Pool(X,label=y,group_id=np.zeros(len(y),'int64'),pairs=graph,pairs_weight=weights,cat_features=cats);prepared=time.monotonic();model=CatBoostRanker(**plan['parameters']);model.fit(pool);trained=time.monotonic();score=model.predict(X.iloc[:2048]);assert np.isfinite(score).all()
        result=dict(status='passed_synthetic_no_official_SAT_or_quality',completed_utc=now(),rows=len(X),features=84,weighted_pairs=int(weights.sum()),unique_pairs=len(graph),global_groups=1,CPU_threads=6,GPU_used=False,official_SAT_values_parsed=False,synthetic_labels=True,quality_scores_computed=False,model_parameters=model.get_all_params(),pair_generation_and_feature_seconds=generated-tick,pool_creation_seconds=prepared-generated,two_tree_fit_seconds=trained-prepared,total_seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,guard_peak_RSS_bytes=peak[0],plan_sha256=sha(ROOT/'research_r67_resource_plan_v2.json'));assert result['peak_RSS_bytes']<=8589934592;dump(ROOT/'research_r67_probe_v2_result.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:pd.read_csv=original;stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

"""Full-row synthetic-label resource compatibility check; no official SAT loading/scoring."""
from research_r32_engine import *
from research_r32_features import ConditionalTabMFeatures
from research_r23_loader import auxiliary
import pandas as pd
from threadpoolctl import threadpool_limits

def main():
    plan=__import__('json').loads((ROOT/'research_r32_resource_plan.json').read_text())
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
    out=ROOT/'research_artifacts/r32_conditional_tabm_hardware';out.mkdir(exist_ok=False)
    peak,stop,watcher=setup(out)
    try:
        columns=list(pd.read_csv(ROOT/'data/train.csv',nrows=0));df=pd.read_csv(ROOT/'data/train.csv',usecols=[c for c in columns if c!='satisfaction']);raw=df.drop(columns='id');y=np.arange(len(df),dtype=np.int64)%2
        with threadpool_limits(6):
            tr=ConditionalTabMFeatures();X=tr.fit_transform(raw,y,auxiliary(df.id));m,rec=fit(X,y,dict(plan['config'],epochs=1),tr.cardinalities,out,'probe')
            q=(X[0][:2048],X[1][:2048]);p=predict(m,q);del m;gc.collect();torch.cuda.empty_cache();loaded=load(out/'probe.pt');again=predict(loaded,q);err=float(np.max(np.abs(p-again)));assert err<1e-7
        assert 0<peak[0]<=1073741824
        result=dict(status='passed_no_quality_score',completed_utc=now(),synthetic_labels=True,official_SAT_column_loaded=False,rows=len(df),feature_width=74,numeric_width=X[0].shape[1],category_columns=22,config=plan['config'],peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_reload_error=err,seconds=rec['seconds'],runner_sha256=sha(__file__))
        dump(out/'result.json',result);print(result,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

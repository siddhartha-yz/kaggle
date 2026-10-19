"""CPU-only shallow forest routing fitted strictly on each outer training fold."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,sha,dump,now
from research_r48_features import covariates,ContextFeatures
from research_r40_worker import watch_cpu
import json,sys,threading,joblib,hashlib,gc,numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

def main(stage,k):
    plan=json.loads((ROOT/'research_r66_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r66_routed_tabpfn_fivefold/result.json').read_text())['eligible']
    base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');labels=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);ids,frame=covariates('train');y=labels.satisfaction.to_numpy('int8');assert np.array_equal(ids,labels.id) and np.array_equal(ids,base['id']) and np.array_equal(y,base['target']);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k)
    out=ROOT/f'research_artifacts/r66_routed_tabpfn_{stage}/fold{k}';out.mkdir(parents=True,exist_ok=True)
    if (out/'routing_manifest.json').exists():
        r=json.loads((out/'routing_manifest.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r66_plan.json') and r['runner_sha256']==sha(__file__)
        for n,h in r['file_sha256'].items():assert sha(out/n)==h,n
        return
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,out/'routing_resource_failure.json'),daemon=True);watcher.start()
    try:
        tr=ContextFeatures().fit(frame.iloc[fi]);joblib.dump(tr,out/'routing_transformer.joblib');X=np.nan_to_num(tr.transform(frame.iloc[fi]),nan=-1.);V=np.nan_to_num(tr.transform(frame.iloc[vi]),nan=-1.);assert X.shape==(len(fi),49) and V.shape==(len(vi),49)
        routers=[];files={'routing_transformer.joblib':sha(out/'routing_transformer.joblib')}
        for m in range(4):
            run=out/f'router{m}';run.mkdir(exist_ok=True);selected=np.random.default_rng(20261066+m).permutation(len(fi))[:int(.6*len(fi))];router=DecisionTreeClassifier(**dict(plan['router_parameters'],random_state=42+m))
            with threadpool_limits(6):router.fit(X[selected],y[fi[selected]]);fit_leaf=router.apply(X);valid_leaf=router.apply(V)
            assert np.array_equal(router.classes_,[0,1]) and router.get_depth()<=3 and router.get_n_leaves()<=8
            joblib.dump(router,run/'router.joblib');np.savez_compressed(run/'indices.npz',outer_train=fi,outer_valid=vi,router_fit=fi[selected],fit_leaf=fit_leaf,valid_leaf=valid_leaf);leaves=[]
            for leaf in sorted(np.unique(valid_leaf)):
                ci_all=fi[fit_leaf==leaf];query=vi[valid_leaf==leaf];assert len(ci_all)>=16384 and not np.intersect1d(ci_all,vi).size;ci=np.random.default_rng(20261066+m).permutation(ci_all)[:4096];assert len(ci)==4096 and not np.intersect1d(ci,query).size
                target=run/f'leaf{leaf}';target.mkdir(exist_ok=True);np.savez_compressed(target/'selection.npz',context_idx=ci,query_idx=query);relative=str((target/'selection.npz').relative_to(out));files[relative]=sha(target/'selection.npz');leaves.append(dict(leaf=int(leaf),context_rows=4096,available_context_rows=len(ci_all),query_rows=len(query),selection_sha256=files[relative]))
            for f in ['router.joblib','indices.npz']:files[str((run/f).relative_to(out))]=sha(run/f)
            routers.append(dict(router=m,leaves=leaves,router_training_rows=len(selected),parameters=router.get_params(),router_used_outer_validation=False));del router;gc.collect()
        r=dict(status='complete',completed_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r66_plan.json'),runner_sha256=sha(__file__),outer_train_rows=len(fi),outer_valid_rows=len(vi),router_transformer_fit_only_outer_train=True,router_input_features=49,routers=routers,file_sha256=files,no_quality_scores=True,peak_RSS_bytes=peak[0]);dump(out/'routing_manifest.json',r);print('ROUTERS_READY_NO_QUALITY',stage,k,len(routers),flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

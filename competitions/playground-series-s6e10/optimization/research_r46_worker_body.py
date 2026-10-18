"""Fixed LGB111; only pretext conditioning set changes, five folds before quality."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,guard
from research_r23_loader import auxiliary,pool
from research_external_ablation import external
from research_r46_features import CohortFeatures
from research_oof import params
from threadpoolctl import threadpool_limits
import json,sys,time,gc,joblib,resource,numpy as np,pandas as pd,lightgbm as lgb

def main(stage):
    plan=json.loads((ROOT/'research_r46_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r46_cohort_lgb111_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert all(np.array_equal(base[n],nn[n]) for n in ['id','target','fold']) and np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y)
    src=ROOT/'research_artifacts/r46_cohort_auxiliary';sr=json.loads((src/'result.json').read_text());z=np.load(src/'train_features.npz');assert sr['status']=='complete' and sr['plan_sha256']==sha(ROOT/'research_r46_plan.json') and np.array_equal(z['id'],df.id) and sha(src/'train_features.npz')==sr['features']['train']['sha256']
    aux=pd.concat([auxiliary(df.id),pd.DataFrame(z['values'],columns=z['columns'])],axis=1);assert aux.shape==(len(df),54);ext,_=external('both',df.id);fold=base['fold'];allpred=np.full(len(df),np.nan);out=ROOT/f'research_artifacts/r46_cohort_lgb111_{stage}';out.mkdir(exist_ok=True)
    for k in sorted(np.unique(fold)):
        k=int(k);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';run.mkdir(exist_ok=True)
        if (run/'result.json').exists():
            rec=json.loads((run/'result.json').read_text());assert rec['runner_sha256']==sha(__file__) and rec['plan_sha256']==sha(ROOT/'research_r46_plan.json')
            if rec['status']=='complete':assert rec['model_sha256']==sha(run/'model.txt');v=np.load(run/'validation.npz');assert np.array_equal(v['id'],df.id.iloc[vi]);allpred[vi]=v['pred'];continue
        guard();tick=time.monotonic();rec=dict(status='training',stage=stage,fold=k,started_utc=now(),train_rows=len(fi),valid_rows=len(vi),parameters=params('stats639'),plan_sha256=sha(ROOT/'research_r46_plan.json'),runner_sha256=sha(__file__),cohort_manifest_sha256=sha(src/'result.json'),cohort_features_sha256=sha(src/'train_features.npz'),outer_validation_used_in_fit=False);dump(run/'result.json',rec);dump(ROOT/'research_r46_state.json',dict(rec,pid=os.getpid(),active_training=True,cpu_threads=6))
        with threadpool_limits(6):
            tr=CohortFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi,:27].reset_index(drop=True));pd.testing.assert_frame_equal(V.iloc[:,:84],C,check_exact=True);del control,C
            model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y[fi]);assert np.array_equal(model.classes_,[0,1]);p=model.predict_proba(V)[:,1];model.booster_.save_model(str(run/'model.txt'));loaded=lgb.Booster(model_file=str(run/'model.txt'));error=float(np.max(np.abs(p-loaded.predict(V,num_threads=6))));assert error<1e-12
        allpred[vi]=p;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p);rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,feature_width=111,prefix84_exactly_equal_control=True,model_sha256=sha(run/'model.txt'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),model_reload_error=error,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(run/'result.json',rec);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True);del model,loaded,X,V,tr;gc.collect()
    assert np.isfinite(allpred).all();tree=base['tree']+.46875*(allpred-base['pred']);candidate=pool(tree,nn['pred']);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=allpred,tree=tree,candidate=candidate,reference_v11=nn['candidate']);dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True))
if __name__=='__main__':main(sys.argv[1])

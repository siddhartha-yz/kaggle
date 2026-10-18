"""Samefixed581-tree residualpriorlearner, addvalidated27targetfreefields only."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,guard
from research_r21_features import AuxiliaryFeatures
from research_r23_loader import auxiliary,pool
from research_external_ablation import external
from research_r40_worker import watch_cpu
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
import json,sys,time,threading,gc,joblib,resource,numpy as np,pandas as pd,lightgbm as lgb

def main(stage):
    plan=json.loads((ROOT/'research_r42_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r42_residual84_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert all(np.array_equal(base[n],nn[n]) for n in ['id','target','fold']) and np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y)
    old=np.load(ROOT/('research_artifacts/residual_prior05_oof_t581/oof.npz' if stage=='fivefold' else 'research_artifacts/r4_baseline_replication3/oof.npz'));assert np.array_equal(old['id'],df.id);oldresid=old['pred'] if stage=='fivefold' else old['components'][:,2]
    fold=base['fold'];ext,_=external('both',df.id);aux=auxiliary(df.id);prior=.5*logit(np.clip(ext.orig_teacher_lgb.to_numpy('float64'),1e-6,1-1e-6));out=ROOT/f'research_artifacts/r42_residual84_{stage}';out.mkdir(exist_ok=True);allpred=np.full(len(df),np.nan);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch_cpu,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        for k in sorted(np.unique(fold)):
            k=int(k);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';run.mkdir(exist_ok=True)
            if (run/'result.json').exists():
                r=json.loads((run/'result.json').read_text());assert r['runner_sha256']==sha(__file__) and r['plan_sha256']==sha(ROOT/'research_r42_plan.json')
                if r['status']=='complete':assert r['model_sha256']==sha(run/'model.txt');v=np.load(run/'validation.npz');assert np.array_equal(v['id'],df.id.iloc[vi]);allpred[vi]=v['pred'];continue
            guard();tick=time.monotonic();rec=dict(status='training',started_utc=now(),stage=stage,fold=k,train_rows=len(fi),valid_rows=len(vi),parameters=plan['parameters'],feature_width=84,prior_scale=.5,prior_clip=[1e-6,1-1e-6],outer_validation_used_in_fit=False,plan_sha256=sha(ROOT/'research_r42_plan.json'),runner_sha256=sha(__file__));dump(run/'result.json',rec);dump(ROOT/'research_r42_state.json',dict(rec,pid=os.getpid(),active_training=True,cpu_threads=6))
            with threadpool_limits(6):
                tr=AuxiliaryFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));pd.testing.assert_frame_equal(V,C,check_exact=True);del control,C;joblib.dump(tr,run/'transformer.joblib')
                legacy_error=None
                if stage=='fivefold':
                    legacy=lgb.Booster(model_file=str(ROOT/f'research_artifacts/residual_prior05_oof_t581/fold{k}/model.txt'));assert legacy.feature_name()==list(V.columns[:57]);lp=expit(legacy.predict(V.iloc[:2048,:57],raw_score=True,num_threads=6)+prior[vi[:2048]]);legacy_error=float(np.max(np.abs(lp-oldresid[vi[:2048]])));assert legacy_error<1e-12;del legacy,lp
                model=lgb.LGBMClassifier(**plan['parameters']);model.fit(X,y[fi],init_score=prior[fi]);assert np.array_equal(model.classes_,[0,1]) and model.booster_.num_trees()==581
                margin=model.booster_.predict(V,raw_score=True,num_threads=6);p=expit(margin+prior[vi]);assert np.isfinite(p).all();model.booster_.save_model(str(run/'model.txt'));loaded=lgb.Booster(model_file=str(run/'model.txt'));again=expit(loaded.predict(V,raw_score=True,num_threads=6)+prior[vi]);error=float(np.max(np.abs(p-again)));assert error<1e-12
            allpred[vi]=p;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p,raw_residual_margin=margin,prior=prior[vi]);rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,model_sha256=sha(run/'model.txt'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),model_reload_error=error,legacy57_replay_error=legacy_error,all84_features_identical_to_certified_control=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert rec['peak_rss_bytes']<=8589934592;dump(run/'result.json',rec);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True);del model,loaded,X,V,tr;gc.collect()
        assert np.isfinite(allpred).all();tree=base['tree']+.1875*(allpred-oldresid);candidate=pool(tree,nn['pred']);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=allpred,tree=tree,candidate=candidate,reference_v11=nn['candidate']);dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True))
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1])

"""Same nestedCPU XGBoost84, fixed equal-probability main-booster averaging; no interim outer scores."""
from research_r4_common import ROOT,now,dump,sha,guard
import os,time,gc,json,sys,joblib,resource
import numpy as np,pandas as pd,xgboost as xgb
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from research_r21_features import AuxiliaryFeatures
from research_external_ablation import external
from research_r23_loader import auxiliary
from research_r15_loader import pool
PARAMS=dict(n_estimators=6000,learning_rate=.03,max_depth=8,min_child_weight=5,reg_lambda=5.,reg_alpha=.05,subsample=.9,colsample_bytree=.5,max_bin=256,objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cpu',enable_categorical=True,max_cat_to_onehot=8,n_jobs=6,random_state=20261024)
class ResourceCallback(xgb.callback.TrainingCallback):
    def __init__(self,path,phase):self.path=path;self.phase=phase;self.start=time.monotonic()
    def after_iteration(self,model,epoch,evals_log):
        if epoch%100==0:
            guard();elapsed=time.monotonic()-self.start;assert elapsed<=1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
            dump(self.path,dict(updated_utc=now(),phase=self.phase,round=epoch+1,seconds=elapsed,pid=os.getpid(),active_training=True,CPU_threads=6));print('ITER',self.phase,epoch+1,elapsed,flush=True)
        return False

def fit(stage):
    assert stage in ('fivefold','threefold')
    plan=json.loads((ROOT/'research_r28_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r28_bag_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');reference=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    assert all(np.array_equal(reference[k],tree[k]) for k in ('id','target','fold')) and np.array_equal(reference['id'],df.id) and np.array_equal(reference['target'],y)
    assert np.max(np.abs(pool(tree['tree'],reference['pred'])-reference['candidate']))<1e-12
    fold=reference['fold'];ext,_=external('both',df.id);aux=auxiliary(df.id);assert aux.shape==(len(df),27)
    out=ROOT/f'research_artifacts/r28_bag_{stage}';out.mkdir(exist_ok=True);predall=np.full(len(y),np.nan)
    for k in sorted(np.unique(fold)):
        k=int(k);guard();fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261024)
        assert not np.intersect1d(it,im).size and not np.intersect1d(fi,vi).size and np.isin(it,fi).all() and np.isin(im,fi).all()
        run=out/f'fold{k}';run.mkdir(exist_ok=True);stamp=run/'result.json'
        if stamp.exists() and json.loads(stamp.read_text()).get('status')=='complete':
            rec=json.loads(stamp.read_text());assert rec['model_sha256']==sha(run/'model.ubj') and rec['runner_sha256']==sha(__file__) and rec['plan_sha256']==sha(ROOT/'research_r28_plan.json')
            z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['idx'],vi);predall[vi]=z['pred'];continue
        np.savez_compressed(run/'indices.npz',outer_train=fi,outer_valid=vi,inner_train=it,inner_meta=im)
        rec=dict(status='inner_preprocessing',stage=stage,fold=k,started_utc=now(),params=PARAMS,runner_sha256=sha(__file__),plan_sha256=sha(ROOT/'research_r28_plan.json'),indices_sha256=sha(run/'indices.npz'),nested_maps_refitted_on_inner_train=True,outer_valid_never_used_for_selection=True)
        dump(stamp,rec);tick=time.monotonic();print('START',stage,k,flush=True)
        with threadpool_limits(6):
            tr=AuxiliaryFeatures();X=tr.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True),aux.iloc[it].reset_index(drop=True));V=tr.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));assert X.shape[1]==84
            joblib.dump(tr,run/'inner_transformer.joblib')
            cfg=dict(PARAMS,early_stopping_rounds=200,callbacks=[ResourceCallback(ROOT/'research_r28_state.json',f'{stage}_fold{k}_inner')]);m=xgb.XGBClassifier(**cfg);m.fit(X,y[it],eval_set=[(V,y[im])],verbose=100);best=int(m.best_iteration)+1
            assert 1<=best<=6000;m.save_model(str(run/'selection_model.ubj'));dump(run/'eval_history.json',m.evals_result())
            rec.update(inner_transformer_sha256=sha(run/'inner_transformer.joblib'),inner_model_sha256=sha(run/'selection_model.ubj'),inner_eval_history_sha256=sha(run/'eval_history.json'),status='outer_refit',selected_rounds=best,inner_best_auc=float(m.best_score),inner_fitted_rounds=m.get_booster().num_boosted_rounds(),selection_at_cap=best>5800);dump(stamp,rec);del m,X,V,tr;gc.collect()
            tr=AuxiliaryFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            cfg=dict(PARAMS,n_estimators=best,callbacks=[ResourceCallback(ROOT/'research_r28_state.json',f'{stage}_fold{k}_outer')]);m=xgb.XGBClassifier(**cfg);m.fit(X,y[fi],verbose=False);assert m.get_booster().num_boosted_rounds()==best and np.array_equal(m.classes_,[0,1]);p=m.predict_proba(V)[:,1];m.save_model(str(run/'model.ubj'))
            loaded=xgb.XGBClassifier(n_jobs=6,device='cpu');loaded.load_model(str(run/'model.ubj'));loaded.get_booster().set_param(dict(nthread=6,device='cpu'));again=loaded.predict_proba(V)[:,1];error=float(np.max(np.abs(p-again)));assert error<1e-7 and np.isfinite(p).all()
        predall[vi]=p;newtree=tree['tree'][vi]+.234375*(p-tree['pred'][vi]);assert ((newtree>=0)&(newtree<=1)).all()
        np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p,candidate=pool(newtree,reference['pred'][vi]))
        rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,feature_width=84,model_sha256=sha(run/'model.ubj'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),model_reload_error=error,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(stamp,rec);print('DONE',stage,k,rec['seconds'],flush=True);del m,loaded,X,V,tr;gc.collect()
    assert np.isfinite(predall).all();newtree=tree['tree']+.234375*(predall-tree['pred']);candidate=pool(newtree,reference['pred'])
    np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,tree=newtree,candidate=candidate,reference_v11=reference['candidate'])
    dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_outer_quality_scoring=True));print('STAGE_COMPLETE',stage,flush=True)
if __name__=='__main__':
    try:fit(sys.argv[1])
    except BaseException as error:dump(ROOT/'research_r28_failure.json',dict(error=repr(error),updated_utc=now()));raise

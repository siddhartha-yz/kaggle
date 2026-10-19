"""Proper nested AUC iteration selection for a global positive-negative PairLogit84 expert."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,guard
from research_r40_worker import watch_cpu
from research_r21_features import AuxiliaryFeatures
from research_r23_loader import auxiliary
from research_external_ablation import external
from research_r67_pairs import pairs,intercept
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker,Pool
from threadpoolctl import threadpool_limits
from scipy.special import expit
import json,sys,threading,time,resource,hashlib,gc,joblib,numpy as np,pandas as pd

def category_strings(frame):
    out=frame.copy();cats=out.select_dtypes('category').columns.tolist()
    for c in cats:out[c]=out[c].astype(str)
    return out,cats

def make_pool(X,y,run,name):
    graph,weights=pairs(y);path=run/f'{name}_pairs.npz';np.savez_compressed(path,pairs=graph,weights=weights)
    pool=Pool(X,label=y,group_id=np.zeros(len(y),'int64'),pairs=graph,pairs_weight=weights,cat_features=[c for c in X if not pd.api.types.is_numeric_dtype(X[c])])
    record=dict(rows=len(y),weighted_pairs=int(weights.sum()),unique_pairs=len(graph),positive_winners_only=True,minimum_weighted_degree=float(np.bincount(graph.ravel(),weights=np.repeat(weights,2),minlength=len(y)).min()),pair_sha256=sha(path),seed=20261067,draws_per_row=8,groups=1)
    del graph,weights;gc.collect();return pool,record

def main(stage,k):
    plan=json.loads((ROOT/'research_r67_plan.json').read_text());assert __import__('catboost').__version__==plan['catboost_version']
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r67_pairlogit84_fivefold/result.json').read_text())['eligible']
    assert json.loads((ROOT/'research_r67_probe_v2_result.json').read_text())['status']=='passed_synthetic_no_official_SAT_or_quality'
    stop=threading.Event();peak=[0];out=ROOT/f'research_artifacts/r67_pairlogit84_{stage}';out.mkdir(exist_ok=True);run=out/f'fold{k}';run.mkdir(exist_ok=True);watcher=threading.Thread(target=watch_cpu,args=(stop,peak,run/'resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        frame=pd.read_csv(ROOT/'data/train.csv');assert frame.satisfaction.dtype==bool;raw=frame.drop(columns=['id','satisfaction']);y=frame.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(base['id'],frame.id) and np.array_equal(base['target'],y);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261024);assert not np.intersect1d(np.r_[it,im],vi).size;ext,_=external('both',frame.id);aux=auxiliary(frame.id)
        if (run/'result.json').exists():
            r=json.loads((run/'result.json').read_text())
            if r['status']=='complete':
                assert r['plan_sha256']==sha(ROOT/'research_r67_plan.json') and r['runner_sha256']==sha(__file__)
                for n,h in r['file_sha256'].items():assert sha(run/n)==h,n
                return
        np.savez_compressed(run/'split.npz',outer_train=fi,outer_valid=vi,inner_train=it,inner_meta=im);guard();record=dict(status='inner_selection',started_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r67_plan.json'),runner_sha256=sha(__file__),split_sha256=sha(run/'split.npz'),train_rows=len(fi),valid_rows=len(vi),outer_validation_used_for_fit_selection_or_calibration=False,CPU_threads=6,GPU_used=False);dump(run/'result.json',record);dump(ROOT/'research_r67_state.json',dict(record,pid=os.getpid(),active_training=True))
        with threadpool_limits(6):
            inner=AuxiliaryFeatures();IX=inner.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True),aux.iloc[it].reset_index(drop=True));IM=inner.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));assert inner.base.n==len(it) and inner.base.te_maps==[] and IX.shape[1]==84;joblib.dump(inner,run/'selection_transformer.joblib');IX,cats=category_strings(IX);IM,mcats=category_strings(IM);assert cats==mcats
            training,tp=make_pool(IX,y[it],run,'selection_train');validation,vp=make_pool(IM,y[im],run,'selection_meta');selected=CatBoostRanker(**plan['selection_parameters']);selected.fit(training,eval_set=validation,early_stopping_rounds=200,use_best_model=True);history=selected.get_evals_result();curve=history['validation']['AUC'];best=int(selected.get_best_iteration())+1;assert 1<=best<=1308 and best==int(np.argmax(curve))+1 and selected.tree_count_==best;selected.save_model(str(run/'selection_model.cbm'));selected_margin=selected.predict(IM,thread_count=6);np.savez_compressed(run/'selection_prediction.npz',idx=im,id=frame.id.iloc[im].to_numpy(),target=y[im],raw_score=selected_margin)
            selection=dict(status='selected',selected_trees=best,max_budget_trees=1308,maximum_reached=best==1308,inner_train_rows=len(it),inner_meta_rows=len(im),inner_split_sha256=sha(run/'split.npz'),transformer_sha256=sha(run/'selection_transformer.joblib'),model_sha256=sha(run/'selection_model.cbm'),prediction_sha256=sha(run/'selection_prediction.npz'),train_pairs=tp,meta_pairs=vp,parameters=plan['selection_parameters'],auc_curve=curve,outer_validation_used=False);dump(run/'selection.json',selection);print('INNER_SELECTED',stage,k,best,flush=True);del selected,inner,training,validation,IX,IM;gc.collect()
            tr=AuxiliaryFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));pd.testing.assert_frame_equal(V,C,check_exact=True);assert tr.base.n==len(fi) and tr.base.te_maps==[];joblib.dump(tr,run/'transformer.joblib');del control,C;X,cats=category_strings(X);V,vcats=category_strings(V);assert cats==vcats and X.shape[1]==84
            training,op=make_pool(X,y[fi],run,'outer_train');cfg=dict(plan['refit_parameters'],iterations=best);record.update(status='outer_refit',selected_trees=best,selection_sha256=sha(run/'selection.json'),parameters=cfg,feature_width=84,prefix84_identical_to_control=True,outer_pairs=op);dump(run/'result.json',record);dump(ROOT/'research_r67_state.json',dict(record,pid=os.getpid(),active_training=True));model=CatBoostRanker(**cfg);model.fit(training);assert model.tree_count_==best and model.feature_names_==list(X) and model.get_cat_feature_indices()==[X.columns.get_loc(c) for c in cats];margin_train=model.predict(X,thread_count=6);bias=intercept(margin_train,y[fi]);calibration=dict(kind='training-only intercept identifies score origin; slope fixed1; no outervalidation',training_rows=len(fi),training_prior=float(y[fi].mean()),bias=bias,mean_probability_residual=float(abs(expit(margin_train+bias).mean()-y[fi].mean())));dump(run/'calibration.json',calibration);np.savez_compressed(run/'calibration_training.npz',idx=fi,id=frame.id.iloc[fi].to_numpy(),target=y[fi],raw_score=margin_train)
            raw_margin=model.predict(V,thread_count=6);pred=expit(raw_margin+bias);assert np.isfinite(raw_margin).all() and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();model.save_model(str(run/'model.cbm'));loaded=CatBoostRanker(thread_count=6);loaded.load_model(str(run/'model.cbm'));error=float(np.max(np.abs(loaded.predict(V.iloc[:2048],thread_count=6)-raw_margin[:2048])));assert error<1e-12;np.savez_compressed(run/'validation.npz',idx=vi,id=frame.id.iloc[vi].to_numpy(),target=y[vi],raw_score=raw_margin,pred=pred)
        files=['split.npz','selection_transformer.joblib','selection_model.cbm','selection_prediction.npz','selection_train_pairs.npz','selection_meta_pairs.npz','selection.json','transformer.joblib','outer_train_pairs.npz','calibration.json','calibration_training.npz','model.cbm','validation.npz'];record.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,native_resolved_parameters=model.get_all_params(),file_sha256={n:sha(run/n) for n in files},model_reload_error=error,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert record['peak_rss_bytes']<=8589934592;dump(run/'result.json',record);print('FOLD_COMPLETE_NO_OUTER_QUALITY',stage,k,best,record['seconds'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except BaseException as e:dump(ROOT/'research_r67_failure.json',dict(status='failed',error=repr(e),utc=now()));raise

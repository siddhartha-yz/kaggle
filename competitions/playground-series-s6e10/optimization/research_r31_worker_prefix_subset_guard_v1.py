"""One fresh GPU process per fixed CB10-84 fit, replacing existing CB10 share."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r31');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
from research_r4_common import now,dump,sha,guard
from research_r31_features import ConditionalCatFeatures
from research_r23_loader import auxiliary
from research_external_ablation import external
from research_r23_guard import watch
from research_r15_loader import pool
import json,time,threading,resource,sys,hashlib,numpy as np,pandas as pd,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split

def main(stage,k):
    os.environ['CUDA_VISIBLE_DEVICES']='0';plan=json.loads((ROOT/'research_r31_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    hw=json.loads((ROOT/'research_artifacts/r31_cb84_hardware/result.json').read_text());assert hw['status']=='passed_no_quality_score' and hw['actual_task_type']=='GPU' and hw['features']==84
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r31_cb84_fivefold/result.json').read_text())['eligible']
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz')
    assert np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y) and all(np.array_equal(base[j],nn[j]) for j in ['id','target','fold'])
    fold=base['fold'];fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);assert len(fi)+len(vi)==len(df) and len(vi)>0 and not np.intersect1d(fi,vi).size
    old=np.load(ROOT/('research_artifacts/cb_deep_oof/raw_oof.npz' if stage=='fivefold' else 'research_artifacts/r4_baseline_replication3/oof.npz'));assert np.array_equal(old['id'],df.id);oldcb=old['pred'] if stage=='fivefold' else old['components'][:,3]
    out=ROOT/f'research_artifacts/r31_cb84_{stage}';out.mkdir(exist_ok=True);run=out/f'fold{k}';run.mkdir(exist_ok=True);record=run/'result.json'
    if record.exists() and json.loads(record.read_text()).get('status')=='complete':
        rec=json.loads(record.read_text());assert rec['model_sha256']==sha(run/'model.cbm') and rec['runner_sha256']==sha(__file__) and rec['plan_sha256']==sha(ROOT/'research_r31_plan.json');return
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);watcher.start()
    try:
        guard();ext,_=external('both',df.id);aux=auxiliary(df.id);tick=time.monotonic()
        it,im=train_test_split(fi,test_size=plan['selection']['inner_valid_fraction'],stratify=y[fi],random_state=plan['selection']['split_seed']);assert not np.intersect1d(it,im).size and np.isin(it,fi).all() and np.isin(im,fi).all()
        np.savez_compressed(run/'inner_split.npz',fit_global=it,valid_global=im)
        inner=ConditionalCatFeatures();IX=inner.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True),aux.iloc[it].reset_index(drop=True));IM=inner.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));joblib.dump(inner,run/'selection_transformer.joblib')
        with threadpool_limits(6):
            selected=CatBoostClassifier(**plan['parameters']);selected.fit(IX,y[it],cat_features=inner.cat_columns,eval_set=(IM,y[im]),use_best_model=True,early_stopping_rounds=plan['selection']['patience'])
        history=selected.get_evals_result()['validation']['AUC'];best=int(selected.get_best_iteration())+1;assert best==int(np.argmax(history))+1 and selected.tree_count_==best and 1<=best<=3000
        selected.save_model(str(run/'selection_model.cbm'));dump(run/'selection.json',dict(status='selected',selected_trees=best,inner_train_rows=len(it),inner_valid_rows=len(im),inner_split_sha256=sha(run/'inner_split.npz'),selection_transformer_sha256=sha(run/'selection_transformer.joblib'),selection_model_sha256=sha(run/'selection_model.cbm'),history=history,parameters=plan['parameters'],selection=plan['selection'],outer_validation_used=False))
        del selected,inner,IX,IM;__import__('gc').collect()
        print('INNER_SELECTED',stage,k,best,flush=True)
        tr=ConditionalCatFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
        if stage=='fivefold':
            control=joblib.load(ROOT/f'research_artifacts/r18_gpu_cb_numeric57_fivefold/fold{k}/transformer.joblib');pd.testing.assert_frame_equal(V.iloc[:64,:57],control.transform(raw.iloc[vi[:64]].reset_index(drop=True),ext.iloc[vi[:64]].reset_index(drop=True)),check_exact=True)
        rec=dict(status='training',started_utc=now(),stage=stage,fold=k,parameters=dict(plan['parameters'],iterations=best),selected_trees=best,selection_sha256=sha(run/'selection.json'),runner_sha256=sha(__file__),plan_sha256=sha(ROOT/'research_r31_plan.json'),train_rows=len(fi),valid_rows=len(vi),train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),feature_width=84,category_columns=tr.cat_columns,outer_validation_used_in_fit=False,prefix57_same_as_cachedGPUcontrol=True if stage=='fivefold' else None);dump(record,rec);dump(ROOT/'research_r31_state.json',dict(rec,pid=os.getpid(),active_training=True))
        with threadpool_limits(6):
            model=CatBoostClassifier(**dict(plan['parameters'],iterations=best));model.fit(X,y[fi],cat_features=tr.cat_columns);assert model.tree_count_==best and np.array_equal(model.classes_,[0,1]) and model.get_all_params()['task_type']=='GPU'
            p=model.predict_proba(V,thread_count=6)[:,1];model.save_model(str(run/'model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(run/'model.cbm'));again=loaded.predict_proba(V,thread_count=6)[:,1];err=float(np.max(np.abs(p-again)));assert err<1e-12
        tree=base['tree'][vi]+.25*(p-oldcb[vi]);assert np.isfinite(p).all() and ((tree>=0)&(tree<=1)).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p,candidate=pool(tree,nn['pred'][vi]))
        rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,model_sha256=sha(run/'model.cbm'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),model_reload_error=err,resolved_parameters=model.get_all_params(),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert 0<peak[0]<=1073741824;dump(record,rec);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except BaseException as error:dump(ROOT/'research_r31_failure.json',dict(error=repr(error),updated_utc=now()));raise

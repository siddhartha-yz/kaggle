"""Inner experts only; reuse exact cached CB8/residual, fit current84 LGB and fixed CB10."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,gc,time,threading,resource,hashlib,joblib,numpy as np,pandas as pd,lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_external_ablation import external
from research_features import DistanceFeatures
from research_r21_features import AuxiliaryFeatures
from research_r23_loader import auxiliary
from research_oof import params

def main(stage,k):
    plan=json.loads((ROOT/'research_r65_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r65_confidence_gate_fivefold/result.json').read_text())['eligible']
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/f'research_r65_tree_{stage}_{k}_resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y);fold=base['fold'];fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);assert not np.intersect1d(np.r_[it,im],vi).size and not np.intersect1d(it,im).size and np.array_equal(np.sort(np.r_[it,im]),fi)
        run=ROOT/f'research_artifacts/r65_confidence_gate_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True);indexpath=run/'inner_indices.npz'
        if indexpath.exists():
            previous=np.load(indexpath);assert all(np.array_equal(previous[n],a) for n,a in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)])
        else:np.savez_compressed(indexpath,outer_train=fi,outer_valid=vi,inner_train=it,inner_meta=im)
        ext,_=external('both',df.id);aux=auxiliary(df.id);teacher=ext.orig_teacher_lgb.to_numpy();out=run/'trees';out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            done=json.loads((out/'result.json').read_text());assert done['status']=='complete' and done['plan_sha256']==sha(ROOT/'research_r65_plan.json') and done['inner_indices_sha256']==sha(run/'inner_indices.npz') and done['prediction_sha256']==sha(out/'meta_predictions.npz')
            for row in done['models'].values():assert sha(ROOT/row['model'])==row['model_sha256']
            assert done['models']['lgb84']['transformer_sha256']==sha(out/'lgb84_transformer.joblib') and done['old57_transformer_sha256']==sha(out/'old57_transformer.joblib');return
        dump(ROOT/'research_r65_tree_state.json',dict(status='inner_tree_fits',stage=stage,fold=k,pid=os.getpid(),active_training=True,updated_utc=now()))
        models={};parts={}
        with threadpool_limits(6):
            tr84=AuxiliaryFeatures();X=tr84.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True),aux.iloc[it].reset_index(drop=True));V=tr84.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));joblib.dump(tr84,out/'lgb84_transformer.joblib');m=lgb.LGBMClassifier(**params('stats639'));m.fit(X,y[it]);parts['lgb84']=m.predict_proba(V)[:,1];m.booster_.save_model(str(out/'lgb84.txt'));loaded=lgb.Booster(model_file=str(out/'lgb84.txt'));error=float(np.max(np.abs(loaded.predict(V,num_threads=6)-parts['lgb84'])));assert error<1e-12;models['lgb84']=dict(model=str((out/'lgb84.txt').relative_to(ROOT)),model_sha256=sha(out/'lgb84.txt'),transformer_sha256=sha(out/'lgb84_transformer.joblib'),CPU_replay_error=error,new_fit=True,trees=639);del m,loaded,X,V,tr84;gc.collect();print('INNER_LGB84_COMPLETE',stage,k,flush=True)
            cache=ROOT/f'research_artifacts/nested_stack_oof/fold{k}'
            if stage=='fivefold':
                z=np.load(cache/'nested_indices.npz');assert all(np.array_equal(z[n],a) for n,a in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)]);oldrec=json.loads((cache/'result.json').read_text());assert oldrec['status']=='complete';tr=joblib.load(cache/'inner_transformer.joblib');assert tr.n==len(it) and tr.te_maps==[]
            else:
                tr=DistanceFeatures('distance_stats_numeric');tr.fit_transform_te(raw.iloc[it].reset_index(drop=True),y[it])
            joblib.dump(tr,out/'old57_transformer.joblib');X=tr.transform_te(raw.iloc[it].reset_index(drop=True));V=tr.transform_te(raw.iloc[im].reset_index(drop=True))
            for c in ext:X[c]=ext[c].to_numpy()[it];V[c]=ext[c].to_numpy()[im]
            assert X.shape[1]==57 and V.shape[1]==57
            for name in ['cb8','residual','cb10']:
                if name in ['cb8','residual'] and stage=='fivefold':
                    oldname='cb_original' if name=='cb8' else 'residual_prior05';file=cache/(oldname+('.cbm' if name=='cb8' else '.txt'));assert oldrec['model_hashes'][oldname]==sha(file);saved=np.load(cache/f'{oldname}_inner_prediction.npz');assert np.array_equal(saved['idx'],im) and np.array_equal(saved['target'],y[im]);p=saved['pred'];reused=True
                    if name=='cb8':
                        model=CatBoostClassifier(thread_count=6);model.load_model(str(file));Q=V.iloc[:2048].copy()
                        for c in Q.select_dtypes('category'):Q[c]=Q[c].astype(str)
                        replay=model.predict_proba(Q,thread_count=6,task_type='CPU')[:,1];del Q
                    else:model=lgb.Booster(model_file=str(file));replay=expit(model.predict(V.iloc[:2048],raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[im[:2048]],1e-6,1-1e-6)))
                    error=float(np.max(np.abs(replay-p[:2048])));assert error<1e-12
                elif name in ['cb8','cb10']:
                    cfg=plan['cb10_parameters'] if name=='cb10' else params('cb1308');CX=X.copy();CV=V.copy();cats=CX.select_dtypes('category').columns.tolist()
                    for c in cats:CX[c]=CX[c].astype(str);CV[c]=CV[c].astype(str)
                    model=CatBoostClassifier(**cfg);model.fit(CX,y[it],cat_features=cats);p=model.predict_proba(CV,thread_count=6)[:,1];file=out/f'{name}.cbm';model.save_model(str(file));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(file));error=float(np.max(np.abs(loaded.predict_proba(CV.iloc[:2048],thread_count=6,task_type='CPU')[:,1]-p[:2048])));assert error<1e-12;reused=False;del CX,CV,loaded
                else:
                    cfg=dict(params('stats639'),n_estimators=581);model=lgb.LGBMClassifier(**cfg);model.fit(X,y[it],init_score=.5*logit(np.clip(teacher[it],1e-6,1-1e-6)));p=expit(model.booster_.predict(V,raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[im],1e-6,1-1e-6)));file=out/'residual.txt';model.booster_.save_model(str(file));loaded=lgb.Booster(model_file=str(file));replay=expit(loaded.predict(V.iloc[:2048],raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[im[:2048]],1e-6,1-1e-6)));error=float(np.max(np.abs(replay-p[:2048])));assert error<1e-12;reused=False;del loaded
                parts[name]=p;models[name]=dict(model=str(file.relative_to(ROOT)),model_sha256=sha(file),reused_exact_inner_cache=reused,CPU_replay_error=error);del model;gc.collect();print('INNER_TREE_COMPLETE',stage,k,name,'reused',reused,flush=True)
        matrix=np.column_stack([parts[n] for n in ['lgb84','cb8','residual','cb10']]);assert np.isfinite(matrix).all() and ((matrix>=0)&(matrix<=1)).all();np.savez_compressed(out/'meta_predictions.npz',idx=im,id=df.id.iloc[im].to_numpy(),target=y[im],components=matrix);result=dict(status='complete',completed_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r65_plan.json'),runner_sha256=sha(__file__),inner_indices_sha256=sha(run/'inner_indices.npz'),prediction_sha256=sha(out/'meta_predictions.npz'),models=models,old57_transformer_sha256=sha(out/'old57_transformer.joblib'),train_rows=len(it),meta_rows=len(im),outer_validation_excluded=True,no_quality_scoring=True,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert result['peak_RSS_bytes']<=8589934592;dump(out/'result.json',result)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

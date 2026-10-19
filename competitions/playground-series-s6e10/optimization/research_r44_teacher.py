"""Sourceonlyteacherselectionandfullfit;neverloadofficialSATlabels."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,guard
from research_r44_source import source_data,normalize
from research_features import DistanceFeatures
from research_r40_worker import watch_cpu
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import json,time,threading,gc,joblib,resource,numpy as np,lightgbm as lgb

def main():
    plan=json.loads((ROOT/'research_r44_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    out=ROOT/'research_artifacts/r44_invistico_teacher';out.mkdir(exist_ok=False);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch_cpu,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        raw,y,train,test,columns,provenance=source_data();fi,vi=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=20261018);np.savez_compressed(out/'source_split.npz',fit_idx=fi,valid_idx=vi);guard();tick=time.monotonic();history={}
        with threadpool_limits(6):
            tr=DistanceFeatures('base');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,out/'selection_transformer.joblib')
            selected=lgb.LGBMClassifier(**plan['teacher_parameters']);selected.fit(X,y[fi],eval_set=[(V,y[vi])],eval_metric='auc',callbacks=[lgb.early_stopping(150,first_metric_only=True,verbose=False),lgb.record_evaluation(history)]);best=int(selected.best_iteration_);curve=history['valid_0']['auc'];assert best==int(np.argmax(curve))+1;selected.booster_.save_model(str(out/'selection_model.txt'),num_iteration=best);p=selected.predict_proba(V)[:,1];assert abs(float(roc_auc_score(y[vi],p))-curve[best-1])<1e-12
            dump(out/'selection.json',dict(status='selected_sourceonly',selected_trees=best,source_AUC=float(roc_auc_score(y[vi],p)),history=history,source_split_sha256=sha(out/'source_split.npz'),selection_model_sha256=sha(out/'selection_model.txt'),selection_transformer_sha256=sha(out/'selection_transformer.joblib'),official_targets_used=False));del selected,tr,X,V;gc.collect()
            full=DistanceFeatures('base');X=full.fit_transform_te(raw,y);assert X.shape[1]==25 and not full.te_maps;model=lgb.LGBMClassifier(**dict(plan['teacher_parameters'],n_estimators=best));model.fit(X,y);assert np.array_equal(model.classes_,[0,1]);model.booster_.save_model(str(out/'model.txt'));joblib.dump(full,out/'transformer.joblib');loaded=lgb.Booster(model_file=str(out/'model.txt'));records={}
            for name,frame in [('train',train),('test',test)]:
                V=full.transform_te(normalize(frame,columns));p=model.predict_proba(V)[:,1];again=loaded.predict(V,num_threads=6);err=float(np.max(np.abs(p-again)));assert err<1e-12 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.savez_compressed(out/f'{name}_features.npz',id=frame.id.to_numpy(),values=p[:,None],columns=np.asarray(['invistico_teacher_lgb']));records[name]=dict(rows=len(frame),features_sha256=sha(out/f'{name}_features.npz'),model_reload_error=err)
        r=dict(status='complete',completed_utc=now(),provenance=provenance,raw_source_features=columns,source_transformed_width=25,source_extra_Online_support_used=False,source_satisfaction_target_only=True,official_satisfaction_loaded=False,teacher_selected_on_source_only=True,selected_trees=best,parameters=dict(plan['teacher_parameters'],n_estimators=best),source_selection_sha256=sha(out/'selection.json'),model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),features=records,plan_sha256=sha(ROOT/'research_r44_plan.json'),seconds=time.monotonic()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(out/'result.json',r);print('TEACHER_COMPLETE',best,r['seconds'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

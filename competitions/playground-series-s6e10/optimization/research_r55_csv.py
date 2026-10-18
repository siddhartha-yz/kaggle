"""Single full proper-inner selection and refit, followed by checked parent-only CSV."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,threading,gc,resource,joblib,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
from research_r4_common import ROOT,now,dump,sha
from research_r55_worker import watch_cpu
from research_r21_features import AuxiliaryFeatures
from research_r23_loader import auxiliary
from research_external_ablation import external
from research_r55_audit import independent_map

def main():
    frozen=json.loads((ROOT/'research_r55_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r55_plan.json').read_text());assert frozen['main_plan_sha256']==sha(ROOT/'research_r55_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**frozen['input_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r55_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_models']==8
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r55_randomized_lgb84_{stage}/result.json').read_text())['eligible']
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r55_full_resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(train)==699635 and train.satisfaction.dtype==bool and len(test)==299844 and np.array_equal(test.id,sample.id) and test.id.is_unique
        raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest);y=train.satisfaction.to_numpy('int8');ext,_=external('both',train.id);zt=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz');assert np.array_equal(zt['id'],test.id);exttest=pd.DataFrame(zt['values'],columns=zt['columns']);aux=auxiliary(train.id);auxtest=auxiliary(test.id,test=True)
        out=ROOT/'research_artifacts/r55_randomized_lgb84_full';out.mkdir(exist_ok=False);it,im=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=20261025);np.savez_compressed(out/'inner_split.npz',fit_global=it,valid_global=im)
        with threadpool_limits(6):
            inner=AuxiliaryFeatures();X=inner.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True),aux.iloc[it].reset_index(drop=True));V=inner.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));joblib.dump(inner,out/'selection_transformer.joblib');history={};selected=lgb.LGBMClassifier(**plan['selection_parameters']);selected.fit(X,y[it],eval_set=[(V,y[im])],eval_metric='auc',callbacks=[lgb.early_stopping(200,first_metric_only=True,verbose=False,min_delta=0.),lgb.record_evaluation(history)])
            assert list(history)==['valid_0'] and list(history['valid_0'])==['auc'];curve=history['valid_0']['auc'];best=int(np.argmax(curve))+1;assert best==selected.best_iteration_ and len(curve)==min(12000,best+200);selected.booster_.save_model(str(out/'selection_model.txt'));sel=dict(selected_trees=best,auc_curve=curve,parameters=plan['selection_parameters'],inner_split_sha256=sha(out/'inner_split.npz'),selection_model_sha256=sha(out/'selection_model.txt'),selection_transformer_sha256=sha(out/'selection_transformer.joblib'),no_test_labels=True);dump(out/'selection.json',sel);del inner,X,V,selected;gc.collect()
            itr,IV=independent_map(out/'selection_transformer.joblib',raw,it,im,ext,aux);reloaded=lgb.Booster(model_file=str(out/'selection_model.txt'));assert reloaded.num_trees()==best;assert abs(roc_auc_score(y[im],reloaded.predict(IV,num_threads=6))-curve[best-1])<1e-12;del itr,IV,reloaded;gc.collect()
            tr=AuxiliaryFeatures();X=tr.fit_transform(raw,y,ext,aux);T=tr.transform(rawtest,exttest,auxtest);control=joblib.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/transformer.joblib');C=control.transform(rawtest,exttest,auxtest);pd.testing.assert_frame_equal(T,C,check_exact=True);del control,C
            parameters=dict(plan['refit_parameters'],n_estimators=best);model=lgb.LGBMClassifier(**parameters);model.fit(X,y);assert np.array_equal(model.classes_,[0,1]);p=model.predict_proba(T)[:,1];model.booster_.save_model(str(out/'model.txt'));joblib.dump(tr,out/'transformer.joblib');loaded=lgb.Booster(model_file=str(out/'model.txt'));assert loaded.num_trees()==best and loaded.num_feature()==84;error=float(np.max(np.abs(p-loaded.predict(T,num_threads=6))));assert error<1e-12
        old=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv');assert np.array_equal(old.id,test.id);candidate=expit(.875*logit(np.clip(old.satisfaction.to_numpy(),1e-6,1-1e-6))+.125*logit(np.clip(p,1e-6,1-1e-6)));assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),pred=p,reference_v11=old.satisfaction.to_numpy(),candidate=candidate)
        csv=ROOT/'submission_randomized_boost_r55_v12.csv';assert not csv.exists();pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g');checked=pd.read_csv(csv);assert list(checked)==list(sample) and len(checked)==299844 and checked.id.is_unique and np.array_equal(checked.id,test.id) and np.isfinite(checked.satisfaction).all() and checked.satisfaction.between(0,1).all();roundtrip=float(np.max(np.abs(checked.satisfaction-candidate)));assert roundtrip<1e-14
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(checked),positive_class='True=1',selected_trees=best,parameters=parameters,model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),selection_sha256=sha(out/'selection.json'),test_components_sha256=sha(out/'test_components.npz'),model_reload_error=error,CSV_roundtrip_error=roundtrip,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,validation=audit['validation'],pre_inference_audit_sha256=sha(ROOT/'research_r55_pre_inference_audit.json'),main_plan_sha256=sha(ROOT/'research_r55_plan.json'),final_inference_plan_sha256=sha(ROOT/'research_r55_final_inference_plan.json'),baseline_CSV_sha256=sha(ROOT/'submission_conditional_neural_r23_v11.csv'),formula=frozen['formula'],parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,limitation=plan['statistics_limitation']);assert result['peak_RSS_bytes']<=8589934592;dump(out/'result.json',result);dump(ROOT/'research_r55_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

"""After both gates and independent audit, full LGB111 and checked parent-only CSV."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,threading,resource,joblib,numpy as np,pandas as pd,lightgbm as lgb
from threadpoolctl import threadpool_limits
from scipy.special import logit,expit
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r62_features import ClassLikelihoodFeatures
from research_r62_loader import auxiliary
from research_r23_loader import pool
from research_external_ablation import external
from research_oof import params

def main():
    frozen=json.loads((ROOT/'research_r62_final_inference_plan.json').read_text());assert frozen['main_plan_sha256']==sha(ROOT/'research_r62_plan.json') and frozen['csv_source_sha256']==sha(__file__)
    for n,h in frozen['input_sha256'].items():assert sha(ROOT/n)==h,n
    plan=json.loads((ROOT/'research_r62_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r62_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_student_models']==8 and audit['verified_source_class_models']==13 and audit['main_plan_sha256']==sha(ROOT/'research_r62_plan.json')
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r62_cohortclass_lgb97_{stage}/result.json').read_text())['eligible']
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(train)==699635 and train.satisfaction.dtype==bool and len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id)
    raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest);y=train.satisfaction.to_numpy('int8');ext,_=external('both',train.id);z=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz');assert np.array_equal(z['id'],test.id);exttest=pd.DataFrame(z['values'],columns=z['columns'])
    aux={name:auxiliary(df.id,test=(name=='test')) for name,df in [('train',train),('test',test)]};assert aux['train'].shape==(699635,40) and aux['test'].shape==(299844,40)
    out=ROOT/'research_artifacts/r62_cohortclass_lgb97_full';out.mkdir(exist_ok=True);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch_cpu,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start();tick=time.monotonic()
    try:
        with threadpool_limits(6):
            transformer=ClassLikelihoodFeatures();X=transformer.fit_transform(raw,y,ext,aux['train']);V=transformer.transform(rawtest,exttest,aux['test']);control=joblib.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/transformer.joblib');C=control.transform(rawtest,exttest,aux['test'].iloc[:,:27]);pd.testing.assert_frame_equal(V.iloc[:,:84],C,check_exact=True)
            model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y);assert np.array_equal(model.classes_,[0,1]);p=model.predict_proba(V)[:,1];model.booster_.save_model(str(out/'model.txt'));joblib.dump(transformer,out/'transformer.joblib');loaded=lgb.Booster(model_file=str(out/'model.txt'));assert loaded.num_trees()==639 and loaded.num_feature()==97;error=float(np.max(np.abs(p-loaded.predict(V,num_threads=6))));assert error<1e-12
        old=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');neural=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(old['id'],test.id) and np.array_equal(neural['id'],test.id) and np.max(np.abs(old['newtree']-neural['v10tree']))<1e-12
        previous=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv');assert np.array_equal(previous.id,test.id) and np.max(np.abs(previous.satisfaction-pool(old['newtree'],neural['newNN'])))<1e-14
        tree=old['newtree']+.46875*(p-old['newLGB']);assert ((tree>=0)&(tree<=1)).all();candidate=pool(tree,neural['newNN']);equivalent=expit(.5*logit(np.clip(tree,1e-6,1-1e-6))+.5*logit(np.clip(neural['newNN'],1e-6,1-1e-6)));assert np.max(np.abs(candidate-equivalent))<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),newLGB=p,newtree=tree,oldtree=old['newtree'],oldNN=neural['newNN'],candidate=candidate)
        csv=ROOT/'submission_source_cohort_class_r62_v12.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id);roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert roundtrip<1e-14
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class=True,model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),test_components_sha256=sha(out/'test_components.npz'),formula='Replaceonly.46875LGB84sharewithLGB97;preservev11CB/residual/CB10andNN74;equalclippedlogitpool.',model_reload_error=error,CSV_roundtrip_error=roundtrip,seconds=time.monotonic()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,validation=audit['validation'],pre_inference_audit_sha256=sha(ROOT/'research_r62_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r62_plan.json'),final_inference_plan_sha256=sha(ROOT/'research_r62_final_inference_plan.json'),parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,limitation=plan['statistics_limitation']);assert result['peak_rss_bytes']<=8589934592;dump(out/'result.json',result);dump(ROOT/'research_r62_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

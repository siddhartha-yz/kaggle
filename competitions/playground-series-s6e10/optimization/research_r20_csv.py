"""Only after two qualified splits and audit: full LGB fit and checked parent CSV."""
import os
for c in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[c]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,resource,joblib,numpy as np,pandas as pd,lightgbm as lgb
from threadpoolctl import threadpool_limits
from scipy.special import logit,expit
from research_r4_common import ROOT,now,dump,sha,guard
from research_oof import params
from research_r19_features import AuxiliaryFeatures
from research_external_ablation import external
from research_r15_loader import pool

def main():
    final=json.loads((ROOT/'research_r20_final_inference_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r20_plan.json')
    for n,h in final['code_sha256'].items():assert sha(ROOT/n)==h
    for n,h in final['inherited_artifact_sha256'].items():assert sha(ROOT/n)==h
    plan=json.loads((ROOT/'research_r20_plan.json').read_text())
    for kind in ('source_sha256','artifact_sha256'):
        for n,h in plan[kind].items():assert sha(ROOT/n)==h
    audit=json.loads((ROOT/'research_r20_pre_inference_audit.json').read_text())
    assert audit['status']=='passed' and audit['verified_new_model_records']==8 and audit['source_rating_models_verified'] in (13,65) and audit['plan_sha256']==sha(ROOT/'research_r20_plan.json')
    variant=audit['variant'];assert variant==json.loads((ROOT/'research_r20_selected.json').read_text())['variant']
    for stage in ('fivefold','threefold'):assert json.loads((ROOT/f'research_artifacts/r20_aux_lgb_{variant}_{stage}/result.json').read_text())['eligible']
    assert sha(ROOT/'data/sample_submission.csv')=='6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert len(train)==699635 and train.satisfaction.dtype==bool and len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id)
    raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest)
    ext,_=external('both',train.id);ext_z=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz');assert np.array_equal(ext_z['id'],test.id)
    ext_test=pd.DataFrame(ext_z['values'],columns=ext_z['columns'])
    src=ROOT/f'research_artifacts/r20_aux_{variant}';sr=json.loads((src/'result.json').read_text());aux={}
    for name,df in [('train',train),('test',test)]:
        z=np.load(src/f'{name}_features.npz');assert np.array_equal(z['id'],df.id) and sha(src/f'{name}_features.npz')==sr[name+'_features_sha256']
        aux[name]=pd.DataFrame(z['values'],columns=z['columns'])
    out=ROOT/f'research_artifacts/r20_aux_lgb_{variant}_full';out.mkdir(exist_ok=True);guard();tick=time.monotonic()
    with threadpool_limits(6):
        tr=AuxiliaryFeatures();X=tr.fit_transform(raw,train.satisfaction.to_numpy('int8'),ext,aux['train']);V=tr.transform(rawtest,ext_test,aux['test'])
        model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,train.satisfaction.to_numpy('int8'));assert np.array_equal(model.classes_,[0,1])
        pred=model.predict_proba(V)[:,1];model.booster_.save_model(str(out/'model.txt'));joblib.dump(tr,out/'transformer.joblib')
        loaded=lgb.Booster(model_file=str(out/'model.txt'));assert loaded.num_trees()==639 and loaded.num_feature()==70
        error=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))));assert error<1e-12
    core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz');nn=np.load(ROOT/'research_artifacts/r4_lean_cv_final/test_components.npz')
    assert np.array_equal(core['id'],test.id) and np.array_equal(nn['id'],test.id)
    oldtree=.46875*core['lgb_original']+.09375*core['cb_original']+.1875*core['residual_prior05']+.25*core['cb_deep']
    assert np.max(np.abs(oldtree-nn['v5']))<1e-12
    newtree=.46875*pred+.09375*core['cb_original']+.1875*core['residual_prior05']+.25*core['cb_deep']
    assert np.max(np.abs(newtree-(oldtree+.46875*(pred-core['lgb_original']))))<1e-12
    assert sha(ROOT/'submission_log_odds_pool_r14_v9.csv')=='55a1cc1805674a4d1419d1c9ff61b64ee35d61f0117782b6c886dd103f9f57c8'
    old=pd.read_csv(ROOT/'submission_log_odds_pool_r14_v9.csv');assert np.array_equal(old.id,test.id) and np.max(np.abs(old.satisfaction-pool(oldtree,nn['realmlp'])))<1e-14
    candidate=pool(newtree,nn['realmlp']);equivalent=expit(.5*logit(np.clip(newtree,1e-6,1-1e-6))+.5*logit(np.clip(nn['realmlp'],1e-6,1-1e-6)))
    assert np.max(np.abs(candidate-equivalent))<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
    np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),newLGB=pred,newtree=newtree,oldtree=oldtree,oldNN=nn['realmlp'],candidate=candidate)
    csv=ROOT/'submission_aux_domain_r20_v10.csv'
    if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
    check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id)
    roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert roundtrip<1e-14
    result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class=True,variant=variant,auxiliary_transductive=(variant=='officialcb'),
        model_sha256=sha(out/'model.txt'),transformer_sha256=sha(out/'transformer.joblib'),test_components_sha256=sha(out/'test_components.npz'),
        family_formula='replaceLGB.46875shareinsidev5tree; equal-logoddspoolwitholdleanCV40',
        model_reload_error=error,CSV_roundtrip_error=roundtrip,seconds=time.monotonic()-tick,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        validation=audit['validation'],pre_inference_audit_sha256=sha(ROOT/'research_r20_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r20_plan.json'),
        formal_used=1,remaining=1,parent_only_submitter=True,actual_top100_verified=False,
        limitation='Historically reusedvalidationlabels; conditionalboundsomitadaptivehistory. No leaderboard result established.')
    dump(out/'result.json',result);dump(ROOT/'research_r20_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
if __name__=='__main__':main()

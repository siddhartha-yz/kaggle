"""Gated full fit and checked CSV; parent exclusively controls Kaggle upload."""
from research_r4_common import ROOT,now,dump,sha,guard
import os,json,time,gc,signal,resource
import numpy as np,pandas as pd,joblib,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_original import safe_original
from research_external_ablation import external
from research_r9_model_features import LinkageFeatures
from research_oof import params

def main():
    audit=json.loads((ROOT/'research_r9_pre_inference_audit.json').read_text())
    assert audit['status']=='passed' and audit['verified_new_models']==8
    assert audit['plan_sha256']==sha(ROOT/'research_r9_plan.json')
    protocol=json.loads((ROOT/'research_r9_plan.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest
    for stage in ['fivefold','threefold']:
        assert json.loads((ROOT/f'research_artifacts/r9_linkage_{stage}/result.json').read_text())['eligible']
    hashes={'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
        'test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6',
        'sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
    for name,digest in hashes.items():assert sha(ROOT/'data'/f'{name}.csv')==digest
    original,train,test,cols,sourceaudit=safe_original()
    assert len(train)==699635 and len(test)==299844 and train.satisfaction.dtype==bool
    source,labels=original[cols],original.satisfaction.to_numpy('int8')
    y=train.satisfaction.to_numpy('int8')
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert np.array_equal(sample.id,test.id) and test.id.is_unique
    out=ROOT/'research_artifacts/r9_linkage_full'
    out.mkdir(exist_ok=True)
    guard()
    ext,_=external('both',train.id)
    tick=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('Frozen full-fit timeout')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(protocol['per_fit_timeout_seconds'])
    try:
        with threadpool_limits(6):
            tr=LinkageFeatures(source,labels)
            X=tr.fit_transform(train[cols].reset_index(drop=True),y,ext)
            model=lgb.LGBMClassifier(**params('stats639'))
            model.fit(X,y)
            assert np.array_equal(model.classes_,[0,1])
            model.booster_.save_model(str(out/'model.txt'))
            joblib.dump(tr,out/'transformer.joblib')
            del X
            gc.collect()
            old_ext=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz')
            assert np.array_equal(old_ext['id'],test.id)
            test_ext=pd.DataFrame(old_ext['values'],columns=old_ext['columns'])
            V=tr.transform(test[cols].reset_index(drop=True),test_ext)
            pred=model.predict_proba(V)[:,1]
            loaded=lgb.Booster(model_file=str(out/'model.txt'))
            error=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))))
            assert error<1e-12
    finally:signal.alarm(0)
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz')
    old=np.load(ROOT/'research_artifacts/r4_lean_cv_final/test_components.npz')
    assert np.array_equal(core['id'],test.id) and np.array_equal(old['id'],test.id)
    v5=(60*core['lgb_original']+12*core['cb_original']+24*core['residual_prior05']+32*core['cb_deep'])/128
    assert np.max(np.abs(v5-old['v5']))<1e-12
    half=.5*v5+.5*old['realmlp']
    assert sha(ROOT/'submission_lean_half_r6_v8.csv')=='e277a97762009cd1ebf739f3fa692ce2ee5e4c1cceff5fab225634f74dd98c5b'
    oldcsv=pd.read_csv(ROOT/'submission_lean_half_r6_v8.csv')
    assert np.array_equal(oldcsv.id,test.id) and np.max(np.abs(oldcsv.satisfaction-half))<1e-14
    candidate=(30*pred+6*core['cb_original']+12*core['residual_prior05']+16*core['cb_deep']+64*old['realmlp'])/128
    assert np.max(np.abs(candidate-(half+(30/128)*(pred-core['lgb_original']))))<1e-12
    assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
    np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),pred=pred,candidate=candidate,reference_half=half)
    csv=ROOT/'submission_exemplar_linkage_r9_v9.csv'
    if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
    check=pd.read_csv(csv)
    assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id)
    roundtrip=float(np.max(np.abs(check.satisfaction-candidate)))
    assert roundtrip<1e-14
    rec=dict(status='complete',completed_utc=now(),csv=csv.name,rows=len(check),bytes=csv.stat().st_size,
        sha256=sha(csv),positive_class=True,model_sha256=sha(out/'model.txt'),
        transformer_sha256=sha(out/'transformer.joblib'),test_components_sha256=sha(out/'test_components.npz'),
        CSV_roundtrip_error=roundtrip,model_reload_error=error,seconds=time.monotonic()-tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_audit=sourceaudit,plan_sha256=sha(ROOT/'research_r9_plan.json'),
        pre_inference_audit_sha256=sha(ROOT/'research_r9_pre_inference_audit.json'),
        fixed_weights={'new_linkage_LGB':30/128,'CB8':6/128,'residual':12/128,'CB10':16/128,'leanCV40':64/128},
        formal_used=1,remaining=1,parent_only_submitter=True,actual_top100_verified=False)
    assert rec['peak_rss_bytes']<protocol['peak_rss_ceiling_bytes']
    dump(out/'result.json',rec)
    dump(ROOT/'research_r9_delivery_audit.json',dict(rec,status='passed'))
    print(json.dumps(rec,indent=2),flush=True)

if __name__=='__main__':main()

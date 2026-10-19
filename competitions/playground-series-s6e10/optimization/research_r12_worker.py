"""Fixed population-profile contrast:57 fields, same CPU LGB639, dual baselines."""
from research_r4_common import ROOT,now,dump,sha,guard,evidence
import os,json,time,gc,resource,signal
import numpy as np,pandas as pd,joblib,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_r5_auto_te import data
from research_r12_population_features import PopulationFeatures
from research_external_ablation import external
from research_oof import params
WEIGHT=30/128

def fit(stage,reference_test,plan):
    guard();df,raw,y,fold,official,old_lgb=data(stage)
    oldpath='r4_lean_gpu_init_oof' if stage=='fivefold' else 'r4_lean_gpu_init_replication3'
    oldnn=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')['pred']
    reference=.5*(official-.25*oldnn)/.75+.5*oldnn
    ext,_=external('both',df.id)
    out=ROOT/'research_artifacts'/f'r12_population_{stage}';out.mkdir(exist_ok=True)
    predall=np.full(len(y),np.nan)
    for k in sorted(np.unique(fold)):
        k=int(k);fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        run=out/f'fold{k}';run.mkdir(exist_ok=True);record=run/'result.json'
        if record.exists() and json.loads(record.read_text()).get('status')=='complete':
            rec=json.loads(record.read_text());assert rec['model_sha256']==sha(run/'model.txt') and rec['runner_sha256']==sha(__file__)
            z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);predall[vi]=z['pred'];continue
        def timeout(signum,frame):raise TimeoutError('R12 fixed per-fold timeout')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
        tick=time.monotonic();rec=dict(status='training',started_utc=now(),stage=stage,fold=k,train_rows=len(fi),valid_rows=len(vi),
            parameters=params('stats639'),runner_sha256=sha(__file__),plan_sha256=sha(ROOT/'research_r12_plan.json'),
            outer_validation_covariates_and_labels_excluded_from_fits=True,official_test_covariates_pooled=True,
            reference_test_rows=len(reference_test),query_self_excluded_for_train_and_test=True,
            pool_target_used=False,train_index_sha256=__import__('hashlib').sha256(fi.tobytes()).hexdigest())
        dump(record,rec);dump(ROOT/'research_r12_state.json',dict(rec,pid=os.getpid(),active_training=True));print('START',stage,k,flush=True)
        try:
            with threadpool_limits(6):
                tr=PopulationFeatures(reference_test)
                X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True))
                V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True))
                assert X.shape[1]==57 and V.shape[1]==57 and tr.base.pool_n==len(fi)+len(reference_test)
                joblib.dump(tr,run/'transformer.joblib')
                model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y[fi]);assert np.array_equal(model.classes_,[0,1])
                pred=model.predict_proba(V)[:,1];model.booster_.save_model(str(run/'model.txt'))
                loaded=lgb.Booster(model_file=str(run/'model.txt'));error=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))))
                assert error<1e-12 and np.isfinite(pred).all()
            predall[vi]=pred;candidate=reference[vi]+WEIGHT*(pred-old_lgb[vi]);assert ((candidate>=0)&(candidate<=1)).all()
            np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],idx=vi,pred=pred,candidate=candidate)
            rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,feature_width=57,
                model_sha256=sha(run/'model.txt'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),
                model_reload_error=error,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            assert rec['peak_rss_bytes']<8*1024**3
            dump(record,rec);print('DONE',stage,k,rec['seconds'],flush=True)
            del tr,X,V,model,loaded;gc.collect()
        except BaseException as error:
            rec.update(status='failed',error=repr(error),completed_utc=now());dump(record,rec);raise
        finally:signal.alarm(0)
    assert np.isfinite(predall).all();candidate=reference+WEIGHT*(predall-old_lgb)
    np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=candidate,reference_half=reference,reference_v7=official,old_lgb=old_lgb)
    control=evidence(y,reference,candidate,fold);direct=evidence(y,official,candidate,fold)
    result=dict(status='complete',completed_utc=now(),stage=stage,vs_private_half=control,vs_official_v7=direct,
        eligible=control['eligible_for_new_partition_replication'] and direct['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'))
    dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    plan=json.loads((ROOT/'research_r12_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(ROOT/'data/test.csv')=='ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'
    test=pd.read_csv(ROOT/'data/test.csv');assert len(test)==299844 and 'satisfaction' not in test
    rawtest=test.drop(columns='id')
    result=fit('fivefold',rawtest,plan)
    if result['eligible']:result=fit('threefold',rawtest,plan)
    state=dict(updated_utc=now(),status='complete',last_stage=result['stage'],eligible_for_candidate=result['stage']=='threefold' and result['eligible'],
        active_training=False,formal_used=1,remaining=1,actual_top100_verified=False,whole_top100_task_completed=False)
    dump(ROOT/'research_r12_result.json',state);dump(ROOT/'research_r12_state.json',state)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump(ROOT/'research_r12_failure.json',dict(status='failed',completed_utc=now(),error=repr(error),active_training=False));raise

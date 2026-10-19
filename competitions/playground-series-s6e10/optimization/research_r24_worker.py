"""Fixed original-auxiliary LGB; finish all outer folds before quality scoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,gc,joblib,resource
import numpy as np,pandas as pd,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,guard
from research_r5_auto_te import data
from research_external_ablation import external
from research_r24_features import UncertaintyFeatures as AuxiliaryFeatures
from research_r15_loader import pool
from research_oof import params

def main():
    variant='officialcb'
    stage='threefold' if '--threefold' in sys.argv else 'fivefold'
    plan=json.loads((ROOT/'research_r24_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest
    src=ROOT/f'research_artifacts/r20_aux_{variant}';sr=json.loads((src/'result.json').read_text())
    assert sr['status']=='complete' and sr['plan_sha256']==sha(ROOT/'research_r20_plan.json') and not sr['satisfaction_used_as_target_or_input'] and not sr['satisfaction_column_loaded']
    assert len(sr['models'])==(13 if variant=='sourcecb' else 65) and sr['source_safe_rows']==129859
    assert all(row['model_sha256']==sha(src/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm') for row in sr['models'])
    guard();df,raw,y,fold,*_=data(stage);ext,_=external('both',df.id)
    z=np.load(src/'train_features.npz');assert sha(src/'train_features.npz')==sr['train_features_sha256'] and np.array_equal(z['id'],df.id)
    aux=pd.DataFrame(z['values'],columns=z['columns']);assert aux.shape==(len(y),13)
    extra_src=ROOT/f'research_artifacts/r21_conditional_{variant}';extra_record=json.loads((extra_src/'result.json').read_text());assert extra_record['status']=='complete' and extra_record['plan_sha256']==sha(ROOT/'research_r21_plan.json')
    extra=np.load(extra_src/'train_features.npz');assert np.array_equal(extra['id'],df.id) and sha(extra_src/'train_features.npz')==extra_record['train_features_sha256']
    aux=pd.concat([aux,pd.DataFrame(extra['values'],columns=extra['columns'])],axis=1);assert aux.shape==(len(y),27)
    uncertainty_src=ROOT/'research_artifacts/r24_uncertainty_officialcb';uncertainty_rec=json.loads((uncertainty_src/'result.json').read_text());assert uncertainty_rec['status']=='complete' and uncertainty_rec['plan_sha256']==sha(ROOT/'research_r24_plan.json')
    u=np.load(uncertainty_src/'train_features.npz');assert np.array_equal(u['id'],df.id) and sha(uncertainty_src/'train_features.npz')==uncertainty_rec['train_features_sha256']
    aux=pd.concat([aux,pd.DataFrame(u['values'],columns=u['columns'])],axis=1);assert aux.shape==(len(y),55)
    reference=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    neural_ref=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');assert np.array_equal(neural_ref['id'],df.id)
    assert all(np.array_equal(neural_ref[k],reference[k]) for k in ('id','target','fold'))
    assert np.max(np.abs(pool(reference['tree'],neural_ref['pred'])-neural_ref['candidate']))<1e-12
    old_lgb=reference['pred']
    assert all(np.array_equal(reference[k],v) for k,v in [('id',df.id),('target',y),('fold',fold)])
    out=ROOT/f'research_artifacts/r24_uncertainty_lgb_{variant}_{stage}';out.mkdir(exist_ok=True);allpred=np.full(len(y),np.nan)
    for k in sorted(np.unique(fold)):
        k=int(k);fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k);run=out/f'fold{k}';run.mkdir(exist_ok=True);record=run/'result.json'
        if record.exists() and json.loads(record.read_text()).get('status')=='complete':
            rec=json.loads(record.read_text());assert rec['runner_sha256']==sha(__file__) and rec['model_sha256']==sha(run/'model.txt')
            pred=np.load(run/'validation.npz');assert np.array_equal(pred['id'],df.id.iloc[vi]);allpred[vi]=pred['pred'];continue
        guard();tick=time.monotonic();rec=dict(status='training',started_utc=now(),variant=variant,stage=stage,fold=k,train_rows=len(fi),valid_rows=len(vi),
            parameters=params('stats639'),plan_sha256=sha(ROOT/'research_r24_plan.json'),runner_sha256=sha(__file__),
            source_manifest_sha256=sha(src/'result.json'),aux_features_sha256=sha(src/'train_features.npz'),probability_features_sha256=sha(extra_src/'train_features.npz'),outer_validation_used_in_fit=False)
        dump(record,rec);dump(ROOT/'research_r24_state.json',dict(rec,pid=os.getpid(),active_training=True,device='cpu',cpu_threads=6))
        with threadpool_limits(6):
            tr=AuxiliaryFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True),aux.iloc[fi].reset_index(drop=True))
            V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y[fi]);assert np.array_equal(model.classes_,[0,1])
            pred=model.predict_proba(V)[:,1];model.booster_.save_model(str(run/'model.txt'))
            loaded=lgb.Booster(model_file=str(run/'model.txt'));error=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))));assert error<1e-12
        assert np.isfinite(pred).all();allpred[vi]=pred
        tree=reference['tree'][vi]+.46875*(pred-old_lgb[vi]);assert ((tree>=0)&(tree<=1)).all()
        np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],idx=vi,pred=pred,candidate=pool(tree,neural_ref['pred'][vi]))
        rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,feature_width=112,model_reload_error=error,
            model_sha256=sha(run/'model.txt'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        dump(record,rec);print('DONE',stage,k,rec['seconds'],flush=True);del model,loaded,X,V,tr;gc.collect()
    assert np.isfinite(allpred).all();tree=reference['tree']+.46875*(allpred-old_lgb);candidate=pool(tree,neural_ref['pred'])
    np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=allpred,tree=tree,candidate=candidate,reference_v11=neural_ref['candidate'],reference_v10=reference['candidate'])
    dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True))
    dump(ROOT/'research_r24_state.json',dict(status='stage_complete',stage=stage,updated_utc=now(),active_training=False))
if __name__=='__main__':main()

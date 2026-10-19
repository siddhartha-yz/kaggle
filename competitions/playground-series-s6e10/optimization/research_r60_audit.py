"""Independent source receipts, old auxiliary hashes, own97field maps and eight CPU replays."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,threading,numpy as np,pandas as pd,joblib,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_features import DistanceFeatures
from research_external_ablation import external
from research_r60_loader import auxiliary
from research_r23_loader import pool
from research_oof import params
def main():
    ap=json.loads((ROOT/'research_r60_audit_plan.json').read_text());plan=json.loads((ROOT/'research_r60_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r60_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['audit_code_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r60_driver_state.json').read_text())['status']=='qualified_for_audit';ga=json.loads((ROOT/'research_r60_generator_audit.json').read_text());assert ga['status']=='passed_no_official_SAT_or_quality' and ga['verified_source_models']==13 and ga['semantic_proof']['prefix84_exact'] and not ga['scope']['official_SAT_values_parsed'] and ga['scope']['source_SAT_used_as_conditioning_input'] and ga['scope']['excluded_exact_source_rows']==21
    src=ROOT/'research_artifacts/r60_original_classconditional';sr=json.loads((src/'result.json').read_text());assert sr['status']=='complete' and sha(src/'result.json')==ga['source_manifest_sha256']
    for j,r in enumerate(sr['models']):assert r['model_sha256']==sha(src/f'rating{j}/model.cbm')
    prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65;oldsrc=ROOT/'research_artifacts/r20_aux_officialcb';old=json.loads((oldsrc/'result.json').read_text());assert len(old['models'])==65 and not old['satisfaction_column_loaded'] and not old['satisfaction_used_as_target_or_input']
    for r in old['models']:assert r['model_sha256']==sha(oldsrc/f"rating{r['rating_index']}_fold{r['auxiliary_fold']}"/'model.cbm')
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r60_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');ext,_=external('both',df.id);aux=auxiliary(df.id);records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r60_sourceclass_lgb97_{stage}';res=json.loads((out/'result.json').read_text());assert res['eligible'] and res['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(df)),y)):fold[vi]=k
            assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and all(np.array_equal(a['fold'],fold) and np.array_equal(a['id'],df.id) and np.array_equal(a['target'],y) for a in [z,base,nn]);pred=np.full(len(df),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r60_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r60_worker_body.py'] and r['parameters']==params('stats639') and r['feature_width']==97 and r['prefix84_exactly_equal_control'] and not r['outer_validation_used_in_fit'] and r['source_manifest_sha256']==sha(src/'result.json') and r['source_features_sha256']==sha(src/'train_features.npz')
                for key,name in [('model_sha256','model.txt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert r[key]==sha(run/name)
                tr=joblib.load(run/'transformer.joblib');own=tr.base.base;assert own.n==len(fi) and own.te_maps==[] and own.columns==list(raw);query=vi[:2048]
                with threadpool_limits(6):
                    fresh=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fi].reset_index(drop=True));fresh.te_maps=[];V=fresh.transform_te(raw.iloc[query].reset_index(drop=True))
                    for c in ext:V[c]=ext[c].to_numpy()[query]
                    for c in aux:V[c]=aux[c].to_numpy('float32')[query]
                    actual=tr.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V,actual,check_exact=True);assert V.shape==(len(query),97);control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query,:27].reset_index(drop=True));pd.testing.assert_frame_equal(V.iloc[:,:84],C,check_exact=True);assert np.array_equal(V.iloc[:,84:].to_numpy(),aux.iloc[query,27:].to_numpy('float32'));model=lgb.Booster(model_file=str(run/'model.txt'));assert model.num_trees()==639 and model.num_feature()==97 and model.feature_name()==list(V);saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]);error=float(np.max(np.abs(model.predict(V,num_threads=6)-saved['pred'][:len(query)])));assert error<1e-12;pred[vi]=saved['pred']
                records.append(dict(stage=stage,fold=k,CPU_replay_error=error,model_sha256=r['model_sha256'],own_maps_reconstructed=True,prefix84_exact=True));del model,tr,V,actual,fresh,control,C,saved;gc.collect()
            tree=base['tree']+.46875*(pred-base['pred']);candidate=pool(tree,nn['pred']);assert np.array_equal(pred,z['pred']) and np.max(np.abs(tree-z['tree']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12 and np.array_equal(z['reference_v11'],nn['candidate']);comparison=evidence(y,nn['candidate'],candidate,fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        result=dict(status='passed',audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r60_plan.json'),audit_plan_sha256=sha(ROOT/'research_r60_audit_plan.json'),verified_source_class_models=13,source_generator_audit_sha256=sha(ROOT/'research_r60_generator_audit.json'),rechecked_old_auxiliary_model_hashes=65,old_auxiliary_prior_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),verified_student_models=8,student_replays=records,validation=validation,positive_class='True=1',CPU_threads=6,peak_RSS_bytes=peak[0],actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r60_pre_inference_audit.json',result);print('R60_INDEPENDENT_AUDIT_PASSED',flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

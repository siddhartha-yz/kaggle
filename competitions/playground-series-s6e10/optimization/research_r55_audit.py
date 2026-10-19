"""Independent split, nested selection, feature maps, saved model and blend audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,threading,gc,hashlib,joblib,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r55_worker import watch_cpu
from research_r21_features import AuxiliaryFeatures
from research_features import DistanceFeatures
from research_r23_loader import auxiliary
from research_external_ablation import external

def independent_map(saved,raw,fit,query,ext,aux):
    tr=joblib.load(saved);assert tr.base.mode=='distance_stats_numeric' and not tr.base.te_columns and tr.base.te_maps==[] and tr.base.n==len(fit)
    fresh=AuxiliaryFeatures();fresh.base=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fit].reset_index(drop=True));fresh.base.te_maps=[];fresh.columns=tr.columns
    V=tr.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True))
    E=fresh.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V,E,check_exact=True);assert V.shape[1]==84
    return tr,V

def main():
    ap=json.loads((ROOT/'research_r55_audit_plan.json').read_text());plan=json.loads((ROOT/'research_r55_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r55_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['audit_code_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r55_driver_state.json').read_text())['status']=='qualified_for_audit'
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r55_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65
        auxrecord=json.loads((ROOT/'research_artifacts/r20_aux_officialcb/result.json').read_text());assert not auxrecord['satisfaction_column_loaded'] and not auxrecord['satisfaction_used_as_target_or_input']
        for rec in auxrecord['models']:assert sha(ROOT/f"research_artifacts/r20_aux_officialcb/rating{rec['rating_index']}_fold{rec['auxiliary_fold']}/model.cbm")==rec['model_sha256']
        df=pd.read_csv(ROOT/'data/train.csv');assert df.satisfaction.dtype==bool and len(df)==699635;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');ext,_=external('both',df.id);aux=auxiliary(df.id)
        validations={};records=[]
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r55_randomized_lgb84_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz')
            fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(df)),y)):fold[vi]=k
            assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold) and all(np.array_equal(z[n],base[n]) for n in ['id','target','fold'])
            pred=np.full(len(df),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261025);run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());sel=json.loads((run/'selection.json').read_text());split=np.load(run/'inner_split.npz')
                assert np.array_equal(split['fit_global'],it) and np.array_equal(split['valid_global'],im) and not np.intersect1d(np.r_[it,im],vi).size and np.array_equal(np.sort(np.r_[it,im]),fi)
                assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r55_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r55_worker.py'] and not r['outer_validation_used_for_fit_or_selection'] and r['prefix84_identical_to_control']
                assert r['selection_sha256']==sha(run/'selection.json') and sel['inner_split_sha256']==sha(run/'inner_split.npz') and sel['selection_transformer_sha256']==sha(run/'selection_transformer.joblib') and sel['selection_model_sha256']==sha(run/'selection_model.txt') and not sel['outer_validation_used']
                best=int(np.argmax(sel['auc_curve']))+1;assert best==sel['selected_trees']==r['selected_trees'] and len(sel['auc_curve'])==min(12000,best+200) and sel['parameters']==plan['selection_parameters'] and sel['selection_rule']==plan['selection']
                assert r['parameters']==dict(plan['refit_parameters'],n_estimators=best)
                with threadpool_limits(6):
                    inner,IM=independent_map(run/'selection_transformer.joblib',raw,it,im,ext,aux);selected=lgb.Booster(model_file=str(run/'selection_model.txt'));assert selected.num_trees()==best and selected.num_feature()==84;inner_auc=roc_auc_score(y[im],selected.predict(IM,num_threads=6));assert abs(inner_auc-sel['auc_curve'][best-1])<1e-12;del inner,IM,selected;gc.collect()
                    query=vi[:2048];tr,V=independent_map(run/'transformer.joblib',raw,fi,query,ext,aux);control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V,C,check_exact=True)
                    assert r['model_sha256']==sha(run/'model.txt') and r['transformer_sha256']==sha(run/'transformer.joblib') and r['prediction_sha256']==sha(run/'validation.npz');model=lgb.Booster(model_file=str(run/'model.txt'));assert model.num_trees()==best and model.num_feature()==84 and model.feature_name()==list(V)
                    saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]);error=float(np.max(np.abs(model.predict(V,num_threads=6)-saved['pred'][:len(query)])));assert error<1e-12;pred[vi]=saved['pred']
                records.append(dict(stage=stage,fold=k,selected_trees=best,inner_auc_replay=inner_auc,CPU_replay_error=error,model_sha256=r['model_sha256'],inner_and_outer_maps_independently_reconstructed=True));del tr,V,control,C,model,saved;gc.collect()
            expected=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));assert np.array_equal(pred,z['pred']) and np.array_equal(expected,z['candidate']) and np.array_equal(base['candidate'],z['reference_v11']);e=evidence(y,base['candidate'],expected,fold);assert e['eligible_for_new_partition_replication'];validations[stage]=e
        result=dict(status='passed',audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r55_plan.json'),audit_plan_sha256=sha(ROOT/'research_r55_audit_plan.json'),models=records,verified_models=8,source_rating_model_hashes_rechecked=65,unchanged_source_generators_prior_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),validation=validations,positive_class='True=1',peak_RSS_bytes=peak[0],actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r55_pre_inference_audit.json',result);print('R55_AUDIT_PASSED',flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

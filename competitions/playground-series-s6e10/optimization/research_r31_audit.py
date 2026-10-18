"""Independent nested selection, train-only maps, CPU replay and tree algebra audit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,evidence
from research_features import DistanceFeatures
from research_external_ablation import external
from research_r23_loader import auxiliary
from research_r15_loader import pool
import json,gc,hashlib,numpy as np,pandas as pd,joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

def maps(actual,raw):
    expected=DistanceFeatures('distance_stats_numeric').fit(raw.reset_index(drop=True))
    pd.testing.assert_series_equal(actual.counts,expected.counts);pd.testing.assert_frame_equal(actual.group_means,expected.group_means);assert actual.global_props==expected.global_props
    for k,v in expected.prop_maps.items():pd.testing.assert_series_equal(actual.prop_maps[k],v)
    for k,v in expected.vocab.items():assert list(actual.vocab[k].categories)==list(v.categories)
    assert not actual.te_maps

def main():
    final=json.loads((ROOT/'research_r31_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r31_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r31_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text())['source_rating_models_verified']==65
    df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');ext,_=external('both',df.id);aux=auxiliary(df.id);assert aux.shape==(len(df),27)
    records=[];checks={}
    for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
        out=ROOT/f'research_artifacts/r31_cb84_{stage}';decision=json.loads((out/'result.json').read_text());assert decision['eligible'];z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz')
        fold=np.full(len(df),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
        assert all(np.array_equal(q['id'],df.id) and np.array_equal(q['target'],y) and np.array_equal(q['fold'],fold) for q in [z,base,nn]);pred=np.full(len(df),np.nan)
        for k in range(K):
            fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261029);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());sel=json.loads((run/'selection.json').read_text());spl=np.load(run/'inner_split.npz')
            assert np.array_equal(spl['fit_global'],it) and np.array_equal(spl['valid_global'],im) and not np.intersect1d(it,im).size and not np.intersect1d(fi,vi).size
            assert rec['status']=='complete' and rec['feature_width']==84 and not rec['outer_validation_used_in_fit'] and rec['runner_sha256']==plan['source_sha256']['research_r31_worker.py'] and rec['plan_sha256']==sha(ROOT/'research_r31_plan.json')
            assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi) and rec['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest()
            best=rec['selected_trees'];assert 1<=best<=3000 and rec['parameters']==dict(plan['parameters'],iterations=best) and rec['selection_sha256']==sha(run/'selection.json')
            assert sel['selected_trees']==best==int(np.argmax(sel['history']))+1 and sel['parameters']==plan['parameters'] and sel['selection']==plan['selection'] and not sel['outer_validation_used']
            for key,name in [('inner_split_sha256','inner_split.npz'),('selection_transformer_sha256','selection_transformer.joblib'),('selection_model_sha256','selection_model.cbm')]:assert sel[key]==sha(run/name)
            inner=joblib.load(run/'selection_transformer.joblib');maps(inner.base,raw.iloc[it]);IM=inner.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));sm=CatBoostClassifier(thread_count=6);sm.load_model(str(run/'selection_model.cbm'));assert sm.tree_count_==best and np.array_equal(sm.classes_,[0,1]) and sm.feature_names_==list(IM)
            with threadpool_limits(6):innerpred=sm.predict_proba(IM,thread_count=6)[:,1]
            inner_auc_error=abs(float(roc_auc_score(y[im],innerpred))-float(sel['history'][best-1]));assert inner_auc_error<1e-6
            del sm,IM,inner;gc.collect()
            for key,name in [('model_sha256','model.cbm'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[key]==sha(run/name)
            tr=joblib.load(run/'transformer.joblib');maps(tr.base,raw.iloc[fi]);V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));assert V.shape==(len(vi),84) and np.array_equal(V.iloc[:,-27:].to_numpy(),aux.iloc[vi].to_numpy('float32'))
            if stage=='fivefold':
                ctl=joblib.load(ROOT/f'research_artifacts/r18_gpu_cb_numeric57_fivefold/fold{k}/transformer.joblib');C=ctl.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));pd.testing.assert_frame_equal(V.iloc[:,:57],C,check_exact=True);oldspl=np.load(ROOT/f'research_artifacts/r18_gpu_cb_numeric57_fivefold/fold{k}/inner_split.npz');assert np.array_equal(oldspl['fit_global'],it) and np.array_equal(oldspl['valid_global'],im);del C,ctl
            model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'));assert model.tree_count_==best and np.array_equal(model.classes_,[0,1]) and model.feature_names_==list(V)
            assert [V.columns[j] for j in model.get_cat_feature_indices()]==tr.cat_columns==rec['category_columns'];assert rec['resolved_parameters']['task_type']=='GPU' and rec['peak_gpu_process_bytes']<=1073741824
            saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi])
            with threadpool_limits(6):replay=model.predict_proba(V.iloc[:2048],thread_count=6)[:,1]
            error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<1e-12;pred[vi]=saved['pred'];records.append(dict(stage=stage,fold=k,selected_trees=best,model_sha256=rec['model_sha256'],CPU_replay_error=error,inner_auc_replay_error=inner_auc_error));del tr,V,model;gc.collect()
        assert np.max(np.abs(pred-z['pred']))<1e-12
        if stage=='fivefold':
            from research_nested_stack import outer_predictions
            core=outer_predictions('oof',np.arange(len(df)),df,fold);deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(deep['id'],df.id);core=np.column_stack([core,deep['pred']])
        else:
            old=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz');assert np.array_equal(old['id'],df.id);core=old['components'][:,:4]
        currenttree=.46875*base['pred']+core[:,1:]@np.array([.09375,.1875,.25]);newtree=.46875*base['pred']+.09375*core[:,1]+.1875*core[:,2]+.25*pred
        assert np.max(np.abs(currenttree-base['tree']))<1e-12 and np.max(np.abs(newtree-z['tree']))<1e-12
        formal=pool(currenttree,nn['pred']);candidate=pool(newtree,nn['pred']);assert np.max(np.abs(formal-nn['candidate']))<1e-12 and np.max(np.abs(formal-z['reference_v11']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12
        checks[stage]=dict(vs_formal_v11=evidence(y,formal,candidate,fold));assert checks[stage]['vs_formal_v11']['eligible_for_new_partition_replication']
    result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r31_plan.json'),verified_new_outer_models=len(records),verified_nested_selection_models=len(records),model_records=records,validation=checks,source_auxiliary_models_reused=65,auxiliary_transductive=True,all_inner_outer_maps_membership_verified=True,prefix57_identical_to_matchedGPUcontrol=True,CPU_predictions_replayed=True,independent_tree_algebra_verified=True,positive_class='True=1',actual_top500_verified=False,limitation='ReusedofficialSATlabels;conditionalboundsomitadaptivehistory.')
    dump(ROOT/'research_r31_pre_inference_audit.json',result);print('AUDIT_PASS',len(records),flush=True)
if __name__=='__main__':main()

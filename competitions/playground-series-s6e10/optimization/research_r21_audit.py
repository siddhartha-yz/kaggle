"""Independent source-feature provenance, fold maps, CPU replay and blend audit."""
import os
for c in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[c]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,joblib,numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold,KFold
from research_r4_common import ROOT,now,dump,sha
from research_r21_check import compare
from research_oof import params
from research_r19_aux_source import RATINGS
from research_r21_conditional import conditional_statistics
from catboost import CatBoostClassifier
from research_original import safe_original
from research_features import DistanceFeatures
from research_external_ablation import external
from research_r15_loader import pool

def main():
    final=json.loads((ROOT/'research_r21_final_inference_plan.json').read_text())
    assert final['main_plan_sha256']==sha(ROOT/'research_r21_plan.json')
    for n,h in final['code_sha256'].items():assert sha(ROOT/n)==h
    for n,h in final['inherited_artifact_sha256'].items():assert sha(ROOT/n)==h
    plan=json.loads((ROOT/'research_r21_plan.json').read_text())
    for kind in ('source_sha256','artifact_sha256'):
        for n,h in plan[kind].items():assert sha(ROOT/n)==h
    variant=json.loads((ROOT/'research_r21_selected.json').read_text())['variant']
    decisions=[json.loads((ROOT/f'research_artifacts/r21_probability_lgb_{v}_fivefold/result.json').read_text()) for v in ('sourcecb','officialcb')]
    eligible=[r for r in decisions if r['eligible']]
    assert variant==max(eligible,key=lambda r:r['vs_private_v9']['pooled']['candidate_auc'])['variant']
    source,df,test,cols,provenance=safe_original()
    assert df.satisfaction.dtype==bool and len(df)==699635 and len(test)==299844
    raw=df[cols];y=df.satisfaction.to_numpy('int8');src=ROOT/f'research_artifacts/r20_aux_{variant}'
    sr=json.loads((src/'result.json').read_text());assert sr['status']=='complete' and not sr['satisfaction_column_loaded'] and not sr['satisfaction_used_as_target_or_input']
    assert sr['source_safe_rows']==129859 and sr['plan_sha256']==sha(ROOT/'research_r20_plan.json')
    assert len(sr['models'])==(13 if variant=='sourcecb' else 65) and sr['transductive']==(variant=='officialcb')
    assert sr['peak_gpu_process_bytes']<=1073741824
    combined=pd.concat([df[cols],test[cols]],ignore_index=True);source=source[cols].copy()
    for frame in (source,combined):
        for c in frame.select_dtypes(exclude='number'):frame[c]=frame[c].fillna('__MISSING__').astype(str)
    ids=np.r_[df.id.to_numpy(),test.id.to_numpy()]
    expected_auxfold=np.full(len(combined),-1,'int8')
    for k,(_,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(combined)):expected_auxfold[vi]=k
    af=np.load(src/'auxiliary_folds.npz');assert np.array_equal(af['id'],ids) and np.array_equal(af['fold'],expected_auxfold) and sha(src/'auxiliary_folds.npz')==sr['auxiliary_fold_sha256']
    aux={};errors=[]
    for name,frame in [('train',df),('test',test)]:
        cache=np.load(src/f'{name}_features.npz');assert np.array_equal(cache['id'],frame.id) and sha(src/f'{name}_features.npz')==sr[name+'_features_sha256']
        aux[name]=pd.DataFrame(cache['values'],columns=cache['columns']);assert aux[name].shape==(len(frame),13)
    allaux=np.concatenate([aux['train'].to_numpy(),aux['test'].to_numpy()])
    probability_src=ROOT/f'research_artifacts/r21_conditional_{variant}'
    probability_record=json.loads((probability_src/'result.json').read_text())
    assert probability_record['status']=='complete' and probability_record['plan_sha256']==sha(ROOT/'research_r21_plan.json')
    assert probability_record['source_manifest_sha256']==sha(src/'result.json') and probability_record['no_new_models_fitted'] and not probability_record['satisfaction_loaded']
    assert probability_record['source_models_used']==len(sr['models'])
    probability_aux={}
    for name,frame in [('train',df),('test',test)]:
        cache=np.load(probability_src/f'{name}_features.npz')
        assert np.array_equal(cache['id'],frame.id) and sha(probability_src/f'{name}_features.npz')==probability_record[name+'_features_sha256']
        probability_aux[name]=pd.DataFrame(cache['values'],columns=cache['columns']);assert probability_aux[name].shape==(len(frame),14)
    probability_all=np.concatenate([probability_aux['train'].to_numpy(),probability_aux['test'].to_numpy()])
    assert np.max(np.abs(probability_all[:,-1]-probability_all[:,:13].mean(axis=1,dtype='float64').astype('float32')))<1e-7

    for row in sr['models']:
        j,k=row['rating_index'],row['auxiliary_fold'];c=RATINGS[j];features=[n for n in cols if n!=c]
        base=source if variant=='sourcecb' else combined
        fi=np.arange(len(source)) if variant=='sourcecb' else np.flatnonzero(expected_auxfold!=k)
        vi=np.arange(len(combined)) if variant=='sourcecb' else np.flatnonzero(expected_auxfold==k)
        if variant=='officialcb':assert not np.intersect1d(fi,vi).size
        assert row['rating']==c and row['feature_columns']==features and row['train_rows']==len(fi) and row['valid_rows']==len(vi)
        assert row['target_excluded_from_inputs'] and 'satisfaction' not in features and 'id' not in features and c not in features
        assert row['parameters']==json.loads((ROOT/'research_r20_plan.json').read_text())['auxiliary_parameters'] and row['resolved_parameters']['task_type']=='GPU'
        assert row['train_index_sha256']==__import__('hashlib').sha256(fi.tobytes()).hexdigest() and row['valid_index_sha256']==__import__('hashlib').sha256(vi.tobytes()).hexdigest()
        assert row['peak_gpu_process_bytes']<=1073741824 and row['model_reload_error']<1e-12
        run=src/f'rating{j}_fold{k}'
        assert sha(run/'model.cbm')==row['model_sha256'] and sha(run/'predictions.npz')==row['prediction_sha256']
        saved=np.load(run/'predictions.npz');assert np.array_equal(saved['id'],ids[vi]) and np.array_equal(saved['idx'],vi) and np.array_equal(saved['pred'],allaux[vi,j])
        model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'))
        assert model.tree_count_==300 and model.feature_names_==features and model.get_all_params()['loss_function']=='MultiClass'
        classes=model.classes_.astype('float32');assert np.array_equal(classes,np.sort(base.iloc[fi][c].unique())) and row['classes']==classes.tolist()
        V=combined.iloc[vi[:2048]][features]
        with threadpool_limits(6):prob=model.predict_proba(V,thread_count=6)
        err=float(np.max(np.abs((prob@classes).astype('float32')-allaux[vi[:2048],j])))
        assert err<1e-6;errors.append(dict(rating=c,auxiliary_fold=k,error=err))
        ev,surprise=conditional_statistics(prob,classes,combined[c].to_numpy()[vi[:2048]])
        probability_error=float(np.max(np.abs(surprise-probability_all[vi[:2048],j])))
        assert probability_error<1e-6
        probability_cache=np.load(probability_src/f'rating{j}_fold{k}.npz')
        assert np.array_equal(probability_cache['id'],ids[vi]) and np.array_equal(probability_cache['idx'],vi) and np.array_equal(probability_cache['values'],probability_all[vi,j])
        row_extra=[v for v in probability_record['records'] if v['rating_index']==j and v['auxiliary_fold']==k]
        assert len(row_extra)==1 and row_extra[0]['source_model_sha256']==row['model_sha256']
        assert row_extra[0]['prediction_sha256']==sha(probability_src/f'rating{j}_fold{k}.npz')
        assert row_extra[0]['observed_values_outside_source_classes']==int((~np.isin(combined[c].to_numpy()[vi],classes)).sum())
        del probability_cache

        del model,V,saved;gc.collect()
    for name in aux:
        aux[name]=pd.concat([aux[name],probability_aux[name]],axis=1);assert aux[name].shape[1]==27
    ext,_=external('both',df.id);models=[];checks={}
    for stage,K,seed,oldpath in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
        out=ROOT/f'research_artifacts/r21_probability_lgb_{variant}_{stage}';decision=json.loads((out/'result.json').read_text());assert decision['eligible']
        z=np.load(out/'oof.npz');official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz');private=np.load(ROOT/f'research_artifacts/r14_log_pool_{stage}/oof.npz')
        assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
        fold=np.full(len(y),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
        assert all(np.array_equal(fold,a['fold']) for a in (z,official,private))
        predall=np.full(len(y),np.nan)
        for k in range(K):
            fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text())
            assert rec['status']=='complete' and rec['feature_width']==84 and not rec['outer_validation_used_in_fit']
            assert rec['plan_sha256']==sha(ROOT/'research_r21_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r21_worker.py']
            assert rec['source_manifest_sha256']==sha(src/'result.json') and rec['aux_features_sha256']==sha(src/'train_features.npz') and rec['probability_features_sha256']==sha(probability_src/'train_features.npz')
            assert rec['parameters']==params('stats639')
            assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi)
            for key,n in [('model_sha256','model.txt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[key]==sha(run/n)
            tr=joblib.load(run/'transformer.joblib');expected=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fi].reset_index(drop=True))
            pd.testing.assert_series_equal(tr.base.counts,expected.counts);pd.testing.assert_frame_equal(tr.base.group_means,expected.group_means)
            assert tr.base.global_props==expected.global_props
            for key,mapping in expected.prop_maps.items():pd.testing.assert_series_equal(tr.base.prop_maps[key],mapping)
            for c,dtype in expected.vocab.items():assert list(tr.base.vocab[c].categories)==list(dtype.categories)
            V=tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True),ext.iloc[vi[:2048]].reset_index(drop=True),aux['train'].iloc[vi[:2048]].reset_index(drop=True))
            assert np.array_equal(V.iloc[:,-27:].to_numpy(),aux['train'].iloc[vi[:2048]].to_numpy())
            model=lgb.Booster(model_file=str(run/'model.txt'));assert model.num_trees()==639 and model.num_feature()==84
            saved=np.load(run/'validation.npz');assert np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]) and np.array_equal(saved['idx'],vi)
            with threadpool_limits(6):replay=model.predict(V,num_threads=6)
            error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<1e-12
            predall[vi]=saved['pred'];models.append(dict(stage=stage,fold=k,model_sha256=rec['model_sha256'],replay_error=error))
            del model,tr,expected,V;gc.collect()
        assert np.max(np.abs(predall-z['pred']))<1e-12
        if stage=='fivefold':
            from research_nested_stack import outer_predictions
            core=outer_predictions('oof',np.arange(len(y)),df,fold);deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz')
            assert np.array_equal(deep['id'],df.id);core=np.column_stack([core,deep['pred']])
        else:
            base=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz');assert np.array_equal(base['id'],df.id) and np.array_equal(base['fold'],fold);core=base['components'][:,:4]
        oldtree=core@np.array([.46875,.09375,.1875,.25]);newtree=.46875*predall+core[:,1:]@np.array([.09375,.1875,.25])
        assert np.max(np.abs(oldtree-private['tree']))<1e-12 and np.max(np.abs(newtree-z['tree']))<1e-12
        candidate=pool(newtree,private['neural']);v9=pool(oldtree,private['neural']);v7=.75*oldtree+.25*private['neural']
        assert np.max(np.abs(v9-z['reference_v9']))<1e-12 and np.max(np.abs(v7-official['candidate']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12
        checks[stage]=dict(vs_v9=compare(y,v9,candidate,fold),vs_v7=compare(y,v7,candidate,fold));assert all(c['eligible_for_new_partition_replication'] for c in checks[stage].values())
    result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r21_plan.json'),variant=variant,source_rating_models_verified=len(sr['models']),auxiliary_transductive=sr['transductive'],source_prediction_replays=errors,
        verified_new_model_records=len(models),model_records=models,validation=checks,independent_tree_reconstruction=True,outer_maps_trainonly=True,positive_class='True=1',actual_top100_verified=False,
        limitation='Historically reused validation labels; conditional bounds omit adaptive history; no Kaggle score established.')
    dump(ROOT/'research_r21_pre_inference_audit.json',result);print('AUDIT_PASS',len(models),flush=True)
if __name__=='__main__':main()

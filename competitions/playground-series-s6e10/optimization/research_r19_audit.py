"""Independent source-feature provenance, fold maps, CPU replay and blend audit."""
import os
for c in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[c]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,joblib,numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
from research_r4_common import ROOT,now,dump,sha
from research_r19_check import compare
from research_oof import params
from research_r19_aux_source import RATINGS,encode
from research_original import safe_original
from research_features import DistanceFeatures
from research_external_ablation import external
from research_r15_loader import pool

def main():
    final=json.loads((ROOT/'research_r19_final_inference_plan.json').read_text())
    assert final['main_plan_sha256']==sha(ROOT/'research_r19_plan.json')
    for n,h in final['code_sha256'].items():assert sha(ROOT/n)==h
    plan=json.loads((ROOT/'research_r19_plan.json').read_text())
    for kind in ('source_sha256','artifact_sha256'):
        for n,h in plan[kind].items():assert sha(ROOT/n)==h
    source,df,test,cols,provenance=safe_original()
    assert df.satisfaction.dtype==bool and len(df)==699635 and len(test)==299844
    raw=df[cols];y=df.satisfaction.to_numpy('int8');src=ROOT/'research_artifacts/r19_original_aux'
    sr=json.loads((src/'result.json').read_text());assert sr['status']=='complete' and not sr['satisfaction_used_as_auxiliary_target_or_input']
    assert sr['source_rows']==129859 and sr['plan_sha256']==sha(ROOT/'research_r19_plan.json')
    assert sr['source_provenance']==provenance and sr['feature_columns']==cols and len(sr['models'])==13
    vocab=joblib.load(src/'vocab.joblib');assert sha(src/'vocab.joblib')==sr['vocab_sha256']
    assert vocab=={c:sorted(source[c].dropna().unique().tolist()) for c in source[cols].select_dtypes(exclude='number')}
    aux={};errors=[]
    for name,frame in [('train',df),('test',test)]:
        cache=np.load(src/f'{name}_features.npz');assert np.array_equal(cache['id'],frame.id) and sha(src/f'{name}_features.npz')==sr[name+'_features_sha256']
        aux[name]=pd.DataFrame(cache['values'],columns=cache['columns']);assert aux[name].shape==(len(frame),13)
    for j,c in enumerate(RATINGS):
        row=sr['models'][j];features=[n for n in cols if n!=c];values=np.sort(source[c].dropna().unique())
        assert row['rating']==c and row['feature_columns']==features and row['classes']==values.tolist()
        assert row['target_excluded_from_inputs'] and 'satisfaction' not in features and 'id' not in features
        assert row['source_rows']==129859 and row['fixed_rounds']==300
        assert row['parameters']==dict(plan['auxiliary_parameters'],num_class=len(values))
        assert sha(src/f'rating{j}.ubj')==row['model_sha256'] and sha(src/f'rating{j}.npz')==row['predictions_sha256']
        model=xgb.Booster();model.load_model(src/f'rating{j}.ubj');model.set_param({'nthread':6,'device':'cpu'})
        assert model.num_boosted_rounds()==300 and model.num_features()==20 and model.feature_names==features
        assert json.loads(model.save_config())['learner']['objective']['name']=='multi:softprob'
        for name,frame in [('train',df),('test',test)]:
            X=encode(frame[cols].iloc[:2048],vocab)[features]
            with threadpool_limits(6):prob=model.predict(xgb.DMatrix(X,nthread=6))
            err=float(np.max(np.abs(prob@values.astype('float32')-aux[name].iloc[:2048,j])))
            assert err<1e-6;errors.append(dict(rating=c,dataset=name,error=err))
        del model;gc.collect()
    ext,_=external('both',df.id);models=[];checks={}
    for stage,K,seed,oldpath in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
        out=ROOT/f'research_artifacts/r19_aux_lgb_{stage}';decision=json.loads((out/'result.json').read_text());assert decision['eligible']
        z=np.load(out/'oof.npz');official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz');private=np.load(ROOT/f'research_artifacts/r14_log_pool_{stage}/oof.npz')
        assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
        fold=np.full(len(y),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
        assert all(np.array_equal(fold,a['fold']) for a in (z,official,private))
        predall=np.full(len(y),np.nan)
        for k in range(K):
            fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text())
            assert rec['status']=='complete' and rec['feature_width']==70 and not rec['outer_validation_used_in_fit']
            assert rec['plan_sha256']==sha(ROOT/'research_r19_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r19_worker.py']
            assert rec['source_manifest_sha256']==sha(src/'result.json') and rec['aux_features_sha256']==sha(src/'train_features.npz')
            assert rec['parameters']==params('stats639')
            assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi)
            for key,n in [('model_sha256','model.txt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[key]==sha(run/n)
            tr=joblib.load(run/'transformer.joblib');expected=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fi].reset_index(drop=True))
            pd.testing.assert_series_equal(tr.base.counts,expected.counts);pd.testing.assert_frame_equal(tr.base.group_means,expected.group_means)
            assert tr.base.global_props==expected.global_props
            for key,mapping in expected.prop_maps.items():pd.testing.assert_series_equal(tr.base.prop_maps[key],mapping)
            for c,dtype in expected.vocab.items():assert list(tr.base.vocab[c].categories)==list(dtype.categories)
            V=tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True),ext.iloc[vi[:2048]].reset_index(drop=True),aux['train'].iloc[vi[:2048]].reset_index(drop=True))
            assert np.array_equal(V.iloc[:,-13:].to_numpy(),aux['train'].iloc[vi[:2048]].to_numpy())
            model=lgb.Booster(model_file=str(run/'model.txt'));assert model.num_trees()==639 and model.num_feature()==70
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
    result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r19_plan.json'),source_rating_models_verified=13,source_prediction_replays=errors,
        verified_new_model_records=len(models),model_records=models,validation=checks,independent_tree_reconstruction=True,outer_maps_trainonly=True,positive_class='True=1',actual_top100_verified=False,
        limitation='Historically reused validation labels; conditional bounds omit adaptive history; no Kaggle score established.')
    dump(ROOT/'research_r19_pre_inference_audit.json',result);print('AUDIT_PASS',len(models),flush=True)
if __name__=='__main__':main()

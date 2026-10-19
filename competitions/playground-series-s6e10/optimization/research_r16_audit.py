"""Independent GPU-CB provenance, exact outer maps, CPU replay and family algebra."""
from research_r4_common import ROOT,now,dump,sha
import json,gc
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_features import DistanceFeatures
from research_external_ablation import external
from research_r15_loader import pool
from research_r16_check import compare

def main():
    finalplan=json.loads((ROOT/'research_r16_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256']==sha(ROOT/'research_r16_plan.json')
    for name,digest in finalplan['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=json.loads((ROOT/'research_r16_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest
    selected=json.loads((ROOT/'research_r16_selected.json').read_text());variant=selected['variant']
    decisions=[json.loads((ROOT/f'research_artifacts/r16_gpu_cb_{v}_fivefold/result.json').read_text()) for v in ['numeric57','nominal58']]
    eligible=[r for r in decisions if r['eligible']]
    assert variant==max(eligible,key=lambda r:r['vs_private_v9']['pooled']['candidate_auc'])['variant']
    df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool
    y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id)
    models,checks=[],{}
    for stage,K,seed,oldpath in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
        out=ROOT/f'research_artifacts/r16_gpu_cb_{variant}_{stage}'
        result=json.loads((out/'result.json').read_text());assert result['eligible']
        z=np.load(out/'oof.npz');official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')
        private=np.load(ROOT/f'research_artifacts/r14_log_pool_{stage}/oof.npz')
        assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
        fold=np.full(len(y),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
        assert np.array_equal(z['fold'],fold) and np.array_equal(fold,private['fold'])
        predall=np.full(len(y),np.nan)
        for k in range(K):
            fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k);run=out/f'fold{k}'
            rec=json.loads((run/'result.json').read_text())
            assert rec['status']=='complete' and rec['parameters']==plan['parameters'] and not rec['outer_validation_used_in_fit']
            assert rec['plan_sha256']==sha(ROOT/'research_r16_plan.json')
            assert rec['runner_sha256']==plan['source_sha256']['research_r16_worker.py']
            assert rec['train_index_sha256']==__import__('hashlib').sha256(fi.tobytes()).hexdigest()
            assert rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<8*1024**3
            for key,name in [('model_sha256','model.cbm'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[key]==sha(run/name)
            tr=joblib.load(run/'transformer.joblib');assert tr.nominal==(variant=='nominal58')
            assert len(tr.output_columns)==(58 if tr.nominal else 57) and len(tr.cat_columns)==(9 if tr.nominal else 8)
            expected=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fi].reset_index(drop=True))
            pd.testing.assert_series_equal(tr.base.counts,expected.counts)
            pd.testing.assert_frame_equal(tr.base.group_means,expected.group_means)
            assert tr.base.global_props==expected.global_props and tr.base.names==expected.names
            for key,mapping in expected.prop_maps.items():pd.testing.assert_series_equal(tr.base.prop_maps[key],mapping)
            for c,dtype in expected.vocab.items():assert list(tr.base.vocab[c].categories)==list(dtype.categories)
            saved=np.load(run/'validation.npz');assert np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi])
            V=tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True),ext.iloc[vi[:2048]].reset_index(drop=True))
            if tr.nominal:assert np.array_equal(V.distance_value_category,raw.iloc[vi[:2048]]['Flight Distance'].astype('int64').astype(str))
            model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'))
            assert model.tree_count_==813 and np.array_equal(model.classes_,[0,1])
            assert list(model.feature_names_)==tr.output_columns
            assert model.get_cat_feature_indices()==[tr.output_columns.index(c) for c in tr.cat_columns]
            resolved=model.get_all_params()
            for key,value in [('task_type','GPU'),('depth',10),('max_ctr_complexity',1),('one_hot_max_size',2),('border_count',254),('boosting_type','Plain'),('bootstrap_type','Bayesian'),('leaf_estimation_iterations',10)]:assert resolved[key]==value
            with threadpool_limits(6):replay=model.predict_proba(V,thread_count=6)[:,1]
            error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<1e-12
            predall[vi]=saved['pred'];models.append(dict(stage=stage,fold=k,model_sha256=rec['model_sha256'],CPU_prediction_replay_error=error))
            del model,tr,V,expected;gc.collect()
        assert np.max(np.abs(predall-z['pred']))<1e-12
        if stage=='fivefold':
            from research_nested_stack import outer_predictions
            core=outer_predictions('oof',np.arange(len(y)),df,fold)
            deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(deep['id'],df.id)
            core=np.column_stack([core,deep['pred']])
        else:
            base=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz')
            assert np.array_equal(base['id'],df.id) and np.array_equal(base['fold'],fold);core=base['components'][:,:4]
        oldtree=core@np.array([.46875,.09375,.1875,.25])
        newtree=core[:,:3]@np.array([.46875,.09375,.1875])+.25*predall
        assert np.max(np.abs(oldtree-private['tree']))<1e-12 and np.max(np.abs(newtree-z['tree']))<1e-12
        oldNN=private['neural'];v9=pool(oldtree,oldNN);v7=.75*oldtree+.25*oldNN;candidate=pool(newtree,oldNN)
        assert np.max(np.abs(v9-z['reference_v9']))<1e-12 and np.max(np.abs(v7-official['candidate']))<1e-12
        assert np.max(np.abs(candidate-z['candidate']))<1e-12
        checks[stage]=dict(vs_v9=compare(y,v9,candidate,fold),vs_v7=compare(y,v7,candidate,fold))
        assert all(value['eligible_for_new_partition_replication'] for value in checks[stage].values())
    result=dict(status='passed',audited_utc=now(),variant=variant,verified_new_model_records=len(models),model_records=models,
        validation=checks,outer_validation_excluded_from_all_fits=True,all_maps_and_category_vocab_trainonly=True,
        all_models813_GPU_and_classes01=True,independent_CPU_replay=True,
        plan_sha256=sha(ROOT/'research_r16_plan.json'),formula='replaceCB10probabilityshare.25insidev5; equal-logodds poolwitholdleanNN',
        actual_top100_verified=False,selection_bias='Historically reused labels; conditional bounds omit adaptive selection.')
    dump(ROOT/'research_r16_pre_inference_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['model_records','validation']},indent=2))
if __name__=='__main__':main()

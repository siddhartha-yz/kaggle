"""Independent rating-pretext label scope, 65 CPU replays, own-fold maps and algebra."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,gc,hashlib,threading,joblib,numpy as np,pandas as pd,lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_r23_loader import auxiliary,pool
from research_external_ablation import external
from research_features import DistanceFeatures

def audit_auxiliary(plan):
    from research_r46_auxiliary import data,RATINGS
    os.environ['CUDA_VISIBLE_DEVICES']=''
    real=pd.read_csv;events=[];paths={(ROOT/'data/train.csv').resolve(),(ROOT/'data/test.csv').resolve()}
    def guarded(path,*args,**kwargs):
        if Path(path).resolve() in paths:
            assert kwargs.get('nrows')==0 or (isinstance(kwargs.get('usecols'),list) and 'satisfaction' not in kwargs['usecols'])
            events.append(dict(file=Path(path).name,header_only=kwargs.get('nrows')==0,columns=kwargs.get('usecols')))
        return real(path,*args,**kwargs)
    pd.read_csv=guarded
    try:ids,raw,X,cats,fold,ntrain=data()
    finally:pd.read_csv=real
    expected=np.full(len(ids),-1,'int8')
    for k,(_,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(np.zeros(len(ids)))):expected[vi]=k
    assert np.array_equal(fold,expected) and not set(RATINGS)&set(X) and len(X.columns)==8 and 'id' not in X and 'satisfaction' not in X
    out=ROOT/'research_artifacts/r46_cohort_auxiliary';record=json.loads((out/'result.json').read_text());assert record['status']=='complete' and len(record['models'])==65
    assert not record['satisfaction_values_parsed'] and not record['satisfaction_used_as_input_or_target'] and record['all_thirteen_ratings_excluded_from_inputs']
    assert record['plan_sha256']==sha(ROOT/'research_r46_plan.json') and record['runner_sha256']==plan['source_sha256']['research_r46_auxiliary.py'] and record['feature_columns']==list(X)
    savedfold=np.load(out/'auxiliary_folds.npz');assert np.array_equal(savedfold['id'],ids) and np.array_equal(savedfold['fold'],fold) and record['auxiliary_fold_sha256']==sha(out/'auxiliary_folds.npz')
    cache={};parts=[]
    columns=[f'cohort_rating_ev_{j}' for j in range(13)]+[f'cohort_rating_surprisal_{j}' for j in range(13)]+['cohort_rating_surprisal_mean']
    for name,lo,hi in [('train',0,ntrain),('test',ntrain,len(raw))]:
        z=np.load(out/f'{name}_features.npz');assert sha(out/f'{name}_features.npz')==record['features'][name]['sha256'] and np.array_equal(z['id'],ids[lo:hi]) and list(z['columns'])==columns and z['values'].shape==(hi-lo,27)
        cache[name]=pd.DataFrame(z['values'],columns=z['columns']);parts.append(z['values'])
    values=np.concatenate(parts);assert np.isfinite(values).all() and np.array_equal(values[:,-1],values[:,13:26].mean(axis=1))
    replays=[];seen=set()
    for row in record['models']:
        j,k=row['rating_index'],row['auxiliary_fold'];assert (j,k) not in seen;seen.add((j,k));fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'rating{j}_fold{k}'
        assert row['rating']==RATINGS[j] and row['feature_columns']==list(X) and row['classes']==np.sort(raw[RATINGS[j]].iloc[fi].unique()).tolist()
        assert row['train_rows']==len(fi) and row['valid_rows']==len(vi) and row['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest() and row['valid_index_sha256']==hashlib.sha256(vi.tobytes()).hexdigest() and not np.intersect1d(fi,vi).size
        assert row['parameters']==plan['auxiliary_parameters'] and row['resolved_parameters']['task_type']=='GPU' and row['peak_gpu_process_bytes']<=1073741824 and row['peak_rss_bytes']<=8589934592
        assert row['model_sha256']==sha(run/'model.cbm') and row['prediction_sha256']==sha(run/'predictions.npz') and row['plan_sha256']==sha(ROOT/'research_r46_plan.json') and row['runner_sha256']==plan['source_sha256']['research_r46_auxiliary.py']
        z=np.load(run/'predictions.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],ids[vi]) and np.array_equal(z['ev'],values[vi,j]) and np.array_equal(z['surprisal'],values[vi,13+j]) and np.array_equal(z['classes'],row['classes'])
        model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'));assert model.feature_names_==list(X) and model.tree_count_==300 and model.get_all_params()['loss_function']=='MultiClass'
        take=np.sort(np.random.default_rng(20261046+j*5+k).choice(len(vi),2048,replace=False));query=vi[take];classes=model.classes_.astype(int);assert np.array_equal(classes,z['classes'])
        with threadpool_limits(6):p=model.predict_proba(X.iloc[query],thread_count=6,task_type='CPU')
        expected_ev=(p*classes[None,:]).sum(axis=1).astype('float32')
        # Match each actual label to its actual class column, not label-as-column-index.
        observed=raw[RATINGS[j]].to_numpy()[query];positions=np.array([int(np.flatnonzero(classes==v)[0]) for v in observed]);expected_s=(-np.log(np.maximum(p[np.arange(len(query)),positions],1e-6))).astype('float32')
        ep=float(np.max(np.abs(p-z['probability'][take])));ee=float(np.max(np.abs(expected_ev-z['ev'][take])));es=float(np.max(np.abs(expected_s-z['surprisal'][take])))
        assert ep<1e-7 and ee<1e-6 and es<1e-6
        replays.append(dict(rating_index=j,auxiliary_fold=k,CPU_probability_error=ep,EV_error=ee,surprisal_error=es));del model,z,p;gc.collect()
    assert seen=={(j,k) for j in range(13) for k in range(5)}
    return dict(verified_models=65,inputs=list(X),target_free_transductive=True,all_rating_inputs_excluded=True,auxiliary_own_fold_excluded=True,guarded_loader_reads=events,official_SAT_values_parsed_by_auxiliary_loader=False,replays=replays,manifest_sha256=sha(out/'result.json')),cache

def main():
    audit_plan=json.loads((ROOT/'research_r46_audit_plan.json').read_text());assert audit_plan['audit_source_sha256']==sha(__file__) and audit_plan['main_plan_sha256']==sha(ROOT/'research_r46_plan.json')
    plan=json.loads((ROOT/'research_r46_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r46_cohort_lgb111_{stage}/result.json').read_text())['eligible']
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r46_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        auxaudit,newaux=audit_auxiliary(plan)
        df=pd.read_csv(ROOT/'data/train.csv');assert df.satisfaction.dtype==bool;y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id);aux=pd.concat([auxiliary(df.id),newaux['train']],axis=1);assert aux.shape==(699635,54)
        models=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r46_cohort_lgb111_{stage}';z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
            assert all(np.array_equal(z[n],base[n]) and np.array_equal(z[n],nn[n]) for n in ['id','target','fold']) and np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold)
            allpred=np.full(len(df),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r46_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r46_worker_body.py'] and rec['feature_width']==111 and not rec['outer_validation_used_in_fit'] and rec['prefix84_exactly_equal_control']
                assert rec['cohort_manifest_sha256']==auxaudit['manifest_sha256'] and rec['cohort_features_sha256']==sha(ROOT/'research_artifacts/r46_cohort_auxiliary/train_features.npz')
                for field,name in [('model_sha256','model.txt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[field]==sha(run/name)
                transformer=joblib.load(run/'transformer.joblib');own=transformer.base.base;expected=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[fi].reset_index(drop=True));assert own.n==len(fi) and own.columns==expected.columns and own.te_maps==[]
                pd.testing.assert_series_equal(own.counts,expected.counts);pd.testing.assert_frame_equal(own.group_means,expected.group_means);assert own.global_props==expected.global_props
                for c,mapping in expected.prop_maps.items():pd.testing.assert_series_equal(own.prop_maps[c],mapping)
                for c,dtype in expected.vocab.items():assert list(own.vocab[c].categories)==list(dtype.categories)
                query=vi[:2048];V=transformer.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query,:27].reset_index(drop=True));pd.testing.assert_frame_equal(V.iloc[:,:84],C,check_exact=True);assert np.array_equal(V.iloc[:,-27:].to_numpy(),aux.iloc[query,27:].to_numpy())
                model=lgb.Booster(model_file=str(run/'model.txt'));assert model.num_trees()==639 and model.num_feature()==111 and model.feature_name()==list(V);saved=np.load(run/'validation.npz');assert np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['idx'],vi) and np.array_equal(saved['target'],y[vi])
                with threadpool_limits(6):replay=model.predict(V,num_threads=6)
                error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<1e-12;allpred[vi]=saved['pred'];models.append(dict(stage=stage,fold=k,CPU_replay_error=error,model_sha256=rec['model_sha256']));del model,transformer,expected,V,C,control,saved;gc.collect()
            tree=base['tree']+.46875*(allpred-base['pred']);candidate=pool(tree,nn['pred']);assert np.max(np.abs(allpred-z['pred']))<1e-12 and np.max(np.abs(tree-z['tree']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(nn['candidate']-z['reference_v11']))<1e-12
            validation[stage]=evidence(y,nn['candidate'],candidate,fold);assert validation[stage]['eligible_for_new_partition_replication']
        result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r46_plan.json'),audit_plan_sha256=sha(ROOT/'research_r46_audit_plan.json'),auxiliary=auxaudit,verified_student_models=8,student_replays=models,own_outer_training_maps_verified=True,prefix84_exact=True,blend_algebra_verified=True,positive_class='True=1',validation=validation,peak_rss_bytes=peak[0],actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r46_pre_inference_audit.json',result);print('R46_AUDIT_PASS',len(models),flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

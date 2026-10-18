"""Independent source/label-scope, context membership, feature maps and blend audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,gc,hashlib,threading,numpy as np,pandas as pd,joblib,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,sha,dump,evidence
from research_r40_worker import watch_cpu
from research_r48_features import covariates
from research_features import DistanceFeatures

def main():
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());auditplan=json.loads((ROOT/'research_r54_audit_plan.json').read_text());assert auditplan['main_plan_sha256']==sha(ROOT/'research_r54_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**auditplan['code_sha256'],**auditplan['input_sha256']}.items():assert sha(ROOT/n)==h,n
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/result.json').read_text())['eligible']
    provenance=json.loads((ROOT/'research_r54_pretrained_provenance.json').read_text());assert provenance['checkpoint_sha256']==sha(ROOT/plan['checkpoint']) and provenance['base_alias_SHA_identical'] and provenance['checkpoint_predates_competition']
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r54_static_resource_failure.json'),daemon=True);watcher.start()
    try:
        # Guard the actual generator reads. Only its later, separate validation audit reads SAT.
        oldread=pd.read_csv;reads=[];paths={(ROOT/'data/train.csv').resolve(),(ROOT/'data/test.csv').resolve()}
        def guarded(path,*args,**kwargs):
            if Path(path).resolve() in paths:
                assert kwargs.get('nrows')==0 or (isinstance(kwargs.get('usecols'),list) and 'satisfaction' not in kwargs['usecols']);reads.append(dict(file=Path(path).name,header_only=kwargs.get('nrows')==0,columns=kwargs.get('usecols')))
            return oldread(path,*args,**kwargs)
        pd.read_csv=guarded
        try:trainids,frame=covariates('train');testids,testframe=covariates('test')
        finally:pd.read_csv=oldread
        assert frame.shape==(699635,49) and testframe.shape==(299844,49) and 'satisfaction' not in frame and 'id' not in frame
        columns=list(frame)[:21];source=ROOT/'research_artifacts/original_teacher_v1';sr=json.loads((source/'result.json').read_text());original_path=ROOT/'research_sources/original_cc0/data.csv';mask_path=ROOT/'research_sources/original_safe_mask.npy';assert sha(original_path)==sr['source_sha256'] and sha(mask_path)==sr['safe_mask_sha256']
        original=pd.read_csv(original_path,usecols=columns+['satisfaction']);assert original.satisfaction.dtype==bool
        def canonical(x):
            x=x[columns].copy()
            for c in x:
                x[c]=x[c].astype('float64') if pd.api.types.is_numeric_dtype(x[c]) else x[c].fillna('__MISSING__').astype(str)
            return x
        orig=canonical(original);query=pd.concat([canonical(frame),canonical(testframe)],ignore_index=True);a=pd.util.hash_pandas_object(orig,index=False).to_numpy();b=pd.util.hash_pandas_object(query,index=False).to_numpy();keep=~np.isin(a,b);assert np.array_equal(keep,np.load(mask_path)) and keep.sum()==129859 and (~keep).sum()==21 and not np.isin(a[keep],b).any()
        pairs=0
        for i in np.flatnonzero(~keep):
            match=np.flatnonzero(b==a[i]);verified=[j for j in match if orig.iloc[i].equals(query.iloc[j])];assert verified;pairs+=len(verified)
        safe=original.loc[keep,columns].reset_index(drop=True);transformer=DistanceFeatures('base').fit(safe);transformer.te_maps=[];teacher=lgb.Booster(model_file=str(source/'model.txt'));assert teacher.num_trees()==sr['full_trees']==475 and teacher.num_feature()==25
        teacher_errors={}
        with threadpool_limits(6):
            for name,ids,raw in [('train',trainids,frame),('test',testids,testframe)]:
                z=np.load(source/f'{name}_features.npz');assert np.array_equal(z['id'],ids);j=list(z['columns']).index('orig_teacher_lgb');take=np.random.default_rng(20261054).choice(len(ids),4096,replace=False);V=transformer.transform_te(raw.iloc[take][columns].reset_index(drop=True));assert teacher.feature_name()==list(V);p=teacher.predict(V,num_threads=6);error=float(np.max(np.abs(p-z['values'][take,j])));assert error<1e-12 and np.array_equal(raw.safe_original_teacher_probability.to_numpy(),z['values'][:,j].astype('float32'));teacher_errors[name]=error
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65
        auxsource=ROOT/'research_artifacts/r20_aux_officialcb';ar=json.loads((auxsource/'result.json').read_text());assert not ar['satisfaction_column_loaded'] and not ar['satisfaction_used_as_target_or_input'] and len(ar['models'])==65
        for r in ar['models']:assert r['model_sha256']==sha(auxsource/f"rating{r['rating_index']}_fold{r['auxiliary_fold']}/model.cbm")
        del orig,query,a,b,original,safe,testframe;gc.collect()
        labels=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);assert np.array_equal(labels.id,trainids) and labels.satisfaction.dtype==bool;y=labels.satisfaction.to_numpy('int8');contexts=[];checks={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}';z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');fold=np.full(len(y),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
            assert np.array_equal(z['id'],trainids) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold) and all(np.array_equal(z[n],base[n]) for n in ['id','target','fold']);pred=np.full(len(y),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);permutation=np.random.default_rng(20261054).permutation(fi);members=[];seen=[]
                for m in range(4):
                    ci=permutation[m*4096:(m+1)*4096];run=out/f'fold{k}/context{m}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r54_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r54_worker.py'] and r['parameters']==plan['classifier_parameters'] and not r['outer_validation_used_for_fit_or_selection'] and r['context_rows']==4096 and r['feature_width']==49 and r['context_reload_error']<1e-6 and r['peak_gpu_process_bytes']<=1073741824 and r['peak_RSS_bytes']<=8589934592
                    for field,name in [('context_sha256','context.npz'),('transformer_sha256','transformer.joblib'),('validation_sha256','validation.npz')]:assert r[field]==sha(run/name)
                    ctx=np.load(run/'context.npz');assert np.array_equal(ctx['idx'],ci) and np.array_equal(ctx['id'],trainids[ci]) and np.array_equal(ctx['target'],y[ci]) and np.array_equal(np.unique(ctx['target']),[0,1]) and not np.intersect1d(ci,vi).size
                    tr=joblib.load(run/'transformer.joblib');assert tr.n==4096 and tr.columns==list(frame) and tr.cats==['Gender','Customer Type','Type of Travel','Class'] and np.array_equal(ctx['categorical_indices'],tr.categorical_indices)
                    for c in tr.cats:assert tr.categories[c]==sorted(frame[c].iloc[ci].fillna('__MISSING__').astype(str).unique())
                    assert np.array_equal(ctx['X'],tr.transform(frame.iloc[ci]),equal_nan=True)
                    saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],trainids[vi]) and np.array_equal(saved['target'],y[vi]);assert saved['pred'].shape==(len(vi),) and np.isfinite(saved['pred']).all() and ((saved['pred']>=0)&(saved['pred']<=1)).all();members.append(saved['pred']);seen.extend(ci);contexts.append(dict(stage=stage,fold=k,context=m,context_sha256=r['context_sha256'],validation_sha256=r['validation_sha256']))
                assert len(set(seen))==16384;pred[vi]=np.mean(members,axis=0)
            candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));assert np.max(np.abs(pred-z['pred']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(base['candidate']-z['reference_v11']))<1e-12;checks[stage]=evidence(y,base['candidate'],candidate,fold);assert checks[stage]['eligible_for_new_partition_replication']
        result=dict(status='passed_static_pending_GPU_replays',audited_utc=now(),plan_sha256=sha(ROOT/'research_r54_plan.json'),audit_plan_sha256=sha(ROOT/'research_r54_audit_plan.json'),contexts=contexts,verified_context_count=32,context_own_labels_and_outer_exclusion_verified=True,four_contexts_disjoint_per_fold=True,context_fitted_vocabulary_verified=True,guarded_feature_generator_reads=reads,official_SAT_parsed_by_feature_generator=False,source_original_safe_rows=129859,source_original_excluded_rows=21,value_verified_excluded_pairs=pairs,source_teacher_CPU_replay_errors=teacher_errors,inherited_rating_models_hash_verified=65,inherited_65_model_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),blend_algebra_verified=True,validation=checks,positive_class='True=1',peak_RSS_bytes=peak[0],actual_top500_verified=False,limitations=plan['statistics_limitation']);dump(ROOT/'research_r54_static_audit.json',result);print('STATIC_AUDIT_PASS',len(contexts),flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

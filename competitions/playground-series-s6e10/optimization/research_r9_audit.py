"""Postqualification audit, independently replaying preprocessing on validation rows."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,hashlib,gc
import numpy as np,pandas as pd,joblib,lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from research_original import safe_original
from research_external_ablation import external
from research_oof import params

def main():
    finalplan=json.loads((ROOT/'research_r9_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256']==sha(ROOT/'research_r9_plan.json')
    for name,digest in finalplan['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=json.loads((ROOT/'research_r9_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    source,df,test,cols,sourceaudit=safe_original()
    raw,y=df[cols],df.satisfaction.to_numpy('int8')
    labels=source.satisfaction.to_numpy('int8')
    ext,_=external('both',df.id)
    models,validation=[],{}
    for stage,K,seed,oldpath in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),
                               ('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
        out=ROOT/'research_artifacts'/f'r9_linkage_{stage}'
        assert json.loads((out/'result.json').read_text())['eligible']
        z=np.load(out/'oof.npz');old=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')
        assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
        fold=np.full(len(y),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
        assert np.array_equal(fold,z['fold']) and np.array_equal(fold,old['fold'])
        predicted=np.full(len(y),np.nan)
        for k in range(K):
            fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text())
            assert rec['status']=='complete' and rec['params']==params('stats639')
            assert rec['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest()
            assert rec['plan_sha256']==sha(ROOT/'research_r9_plan.json')
            assert rec['runner_sha256']==plan['source_sha256']['research_r9_worker.py']
            for key,path in [('model_sha256','model.txt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:
                assert rec[key]==sha(run/path)
            tr=joblib.load(run/'transformer.joblib');link=tr.linkage
            assert np.array_equal(tr.source_labels,labels) and tr.source_raw is None
            assert link.n_train==len(fi) and link.fit_summary['train_rows']==len(fi)
            assert not link.fit_summary['official_target_used'] and not link.fit_summary['source_target_used_for_EM']
            assert link.sample_rows==100000 and link.iterations==10 and link.seed==20261026
            assert np.min(np.diff([a['penalized_log_likelihood'] for a in link.trace]))>-1e-4
            assert link.fd_counts.sort_index().equals(raw.iloc[fi]['Flight Distance'].value_counts().sort_index())
            codes=link._encode(raw.iloc[fi]);sourcecodes=link._encode(source[cols])
            assert np.array_equal(sourcecodes,link.source_codes)
            for j,c in enumerate(link.columns):
                union=set(pd.concat([raw.iloc[fi][c],source[c]],ignore_index=True).fillna('__MISSING__').astype(str))
                assert set(link.vocab[j])==union
                expected=pd.Series(codes[:,j]).value_counts().sort_index()
                assert expected.equals(link.global_counts[j].sort_index())
                joint=pd.DataFrame({'fd':raw.iloc[fi]['Flight Distance'].to_numpy(),'value':codes[:,j]}).groupby(['fd','value'],sort=False).size()
                assert joint.sort_index().equals(link.joint_counts[j].sort_index())
            for fd,idx in link.source_groups.items():assert np.all(source.iloc[idx]['Flight Distance'].to_numpy()==fd)
            assert np.array_equal(np.sort(np.concatenate(list(link.source_groups.values()))),np.arange(len(source)))
            saved=np.load(run/'validation.npz')
            assert np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi])
            with threadpool_limits(6):
                V=tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True),ext.iloc[vi[:2048]].reset_index(drop=True))
                model=lgb.Booster(model_file=str(run/'model.txt'))
                assert model.num_trees()==639 and model.num_feature()==64
                p=model.predict(V,num_threads=6)
            err=float(np.max(np.abs(p-saved['pred'][:2048])))
            assert err<1e-12
            predicted[vi]=saved['pred']
            models.append(dict(stage=stage,fold=k,model_sha256=rec['model_sha256'],preprocessing_replay_error=err))
            del tr,link,codes,sourcecodes,model,V
            gc.collect()
        assert np.max(np.abs(predicted-z['pred']))<1e-12
        if stage=='fivefold':
            from research_nested_stack import outer_predictions
            core=outer_predictions('oof',np.arange(len(df)),df,fold)
            deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz')
            assert np.array_equal(deep['id'],df.id)
            core=np.column_stack([core,deep['pred']])
        else:
            base=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz')
            assert np.array_equal(base['id'],df.id) and np.array_equal(base['fold'],fold)
            core=base['components'][:,:4]
        v7=(core@np.array([45,9,18,24])+32*old['pred'])/128
        half=(core@np.array([30,6,12,16])+64*old['pred'])/128
        candidate=(30*predicted+core[:,1:]@np.array([6,12,16])+64*old['pred'])/128
        assert np.max(np.abs(v7-z['reference_v7']))<1e-12
        assert np.max(np.abs(half-z['reference_half']))<1e-12
        assert np.max(np.abs(candidate-z['candidate']))<1e-12
        validation[stage]=dict(vs_official_v7=evidence(y,v7,candidate,fold),vs_private_half=evidence(y,half,candidate,fold))
        assert all(v['eligible_for_new_partition_replication'] for v in validation[stage].values())
    result=dict(status='passed',audited_utc=now(),source_audit=sourceaudit,
        model_records=models,validation=validation,verified_new_models=len(models),
        supervised_fit_outer_validation_excluded=True,EM_fit_official_and_source_targets_excluded=True,
        exact_21_source_matches_excluded=True,matching_vocabulary_and_backgrounds_outertrain_plus_source_only=True,
        plan_sha256=sha(ROOT/'research_r9_plan.json'),actual_top100_verified=False,
        selection_bias='Reused official validation labels; conditional bounds do not cover prior adaptive selection.')
    dump(ROOT/'research_r9_pre_inference_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_records','validation')},indent=2))

if __name__=='__main__':main()

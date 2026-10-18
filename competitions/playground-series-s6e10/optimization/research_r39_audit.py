"""Independent32->128membership, unchangedmaps, seeds, epochs, CPU replay and uniformity."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,cloudpickle,joblib,numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r23_loader import auxiliary,pool
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures

def maps(actual,control,raw,y):
    assert actual.columns==control.columns and actual.cats==control.cats and actual.numeric==control.numeric and actual.output_columns==control.output_columns and actual.aux_columns==control.aux_columns
    assert len(actual.output_columns)==74 and len(actual.aux_columns)==27
    for c in control.counts:
        pd.testing.assert_series_equal(actual.counts[c],control.counts[c]);pd.testing.assert_series_equal(actual.counts[c].sort_index(),raw[c].value_counts().sort_index())
    assert actual.encoder.cv==5 and actual.encoder.random_state==42 and actual.encoder.target_type=='binary' and abs(actual.encoder.target_mean_-y.mean())<1e-15 and actual.encoder.target_mean_==control.encoder.target_mean_
    keys=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.reset_index(drop=True),cols) for j,cols in enumerate(COMBOS)})
    for j,c in enumerate(keys):assert set(actual.encoder.categories_[j])==set(keys[c])
    for a,b in zip(actual.encoder.categories_,control.encoder.categories_):assert np.array_equal(a,b)
    for a,b in zip(actual.encoder.encodings_,control.encoder.encodings_):assert np.array_equal(a,b)

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    final=json.loads((ROOT/'research_r39_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r39_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r39_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    a=ROOT/'research_artifacts/r39_uniform_lean_fivefold/result.json';b=ROOT/'research_artifacts/r39_uniform_fivefold/result.json';assert a.resolve()==b.resolve() and sha(a)==sha(b) and json.loads(a.read_text())['eligible']
    prior=json.loads((ROOT/'research_r23_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['verified_model_records']==8 and prior['verified_native_neural_members']==256
    assert json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text())['source_rating_models_verified']==65
    df=pd.read_csv(ROOT/'data/train.csv');assert df.satisfaction.dtype==bool and len(df)==699635;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');aux=auxiliary(df.id);records=[];checks={}
    with threadpool_limits(6):
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r39_uniform_{stage}';assert json.loads((out/'result.json').read_text())['eligible'];z=np.load(out/'oof.npz');old=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
            assert all(np.array_equal(q['id'],df.id) and np.array_equal(q['target'],y) and np.array_equal(q['fold'],fold) for q in [z,old,tree]);blocks=[old['pred']]
            for network_seed in [43,44,45]:
                prediction=np.full(len(df),np.nan)
                for k in range(K):
                    fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=ROOT/f'research_artifacts/r39_seed{network_seed}_lean_{stage}/fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['seed']==network_seed and r['config']==dict(plan['config'],random_state=network_seed) and r['prefix74_identical_to_control']
                    assert r['plan_sha256']==sha(ROOT/'research_r39_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r39_worker.py'] and r['model_sha256']==sha(run/'model.pkl') and r['transformer_sha256']==sha(run/'transformer.joblib')
                    assert r['train_rows']==len(fi) and r['valid_rows']==len(vi) and not r['validation_used_for_fit_or_epoch_selection'] and r['observed_batch_devices']==['cuda:0'] and len(r['epochs'])==3 and r['feature_width']==74
                    assert r['peak_gpu_process_bytes']<=1073741824 and r['peak_rss_bytes']<=8589934592 and all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=778043392 for e in r['epochs'])
                    tr=joblib.load(run/'transformer.joblib');control=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');maps(tr,control,raw.iloc[fi],y[fi]);V=tr.transform(raw.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));pd.testing.assert_frame_equal(V,control.transform(raw.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True)),check_exact=True)
                    v=np.load(run/'validation.npz');assert np.array_equal(v['idx'],vi) and np.array_equal(v['id'],df.id.iloc[vi]) and np.array_equal(v['target'],y[vi])
                    with open(run/'model.pkl','rb') as f:m=cloudpickle.load(f)
                    m.to('cpu');module=m.cv_alg_interface_.model;module.creator.device_info='cpu';assert np.array_equal(m.classes_,[0,1]) and module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3 and all(p.device.type=='cpu' for p in module.parameters())
                    replay=m.predict_proba(V.iloc[:2048])[:,1];err=float(np.max(np.abs(replay-v['pred'][:2048])));assert err<2e-6;prediction[vi]=v['pred'];records.append(dict(stage=stage,fold=k,seed=network_seed,model_sha256=r['model_sha256'],CPU_replay_error=err,native_members=32));del m,module,tr,control,V;gc.collect()
                assert np.max(np.abs(prediction-z[f'seed{network_seed}']))<1e-12;blocks.append(prediction)
            ensemble=np.mean(blocks,axis=0);assert np.max(np.abs(ensemble-z['pred']))<1e-12 and np.max(np.abs(blocks[0]-z['seed42']))<1e-12
            formal=pool(tree['tree'],old['pred']);candidate=pool(tree['tree'],ensemble);assert np.max(np.abs(formal-old['candidate']))<1e-12 and np.max(np.abs(formal-z['reference_v11']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12
            checks[stage]=evidence(y,formal,candidate,fold);assert checks[stage]['eligible_for_new_partition_replication']
    result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r39_plan.json'),new_model_records_verified=len(records),new_native_members_verified=32*len(records),old_seed42_models_reused=8,all_models_per_fold=4,native_members_per_fold=128,all_new_models_CPU_replayed=True,maps_identical_to_owntraining_certified_control=True,seed42_not_retrained=True,seed_selection=False,uniform_fourblock_algebra_verified=True,positive_class='True=1',alias_same_canonical_result_verified=True,model_records=records,validation=checks,final_test_members=640,actual_top500_verified=False,limitation='ReusedofficialSATvalidationlabels;conditionalboundsomitadaptivehistory.')
    assert len(records)==24;dump(ROOT/'research_r39_pre_inference_audit.json',result);print('AUDIT_PASS',len(records),flush=True)
if __name__=='__main__':main()

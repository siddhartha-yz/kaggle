"""CPU metadata/membership audit; GPU replays are required before final inference."""
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r15_loader import pool
from research_r36_check import combine
from sklearn.model_selection import StratifiedKFold
import json,numpy as np,pandas as pd,sys

def main():
    final=json.loads((ROOT/'research_r36_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r36_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r36_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;y=df.satisfaction.to_numpy('int8');records=[];checks={}
    assert plan['parity_evidence']['same_schedule_old_new_max_probability_error']==0 and plan['parity_evidence']['A_B_A_error']==0
    assert json.loads((ROOT/'research_r23_pre_inference_audit.json').read_text())['verified_model_records']==8
    assert json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text())['source_rating_models_verified']==65
    for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
        out=ROOT/f'research_artifacts/r36_tabicl48_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'];z=np.load(out/'oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(df,y)):fold[vi]=k
        assert all(np.array_equal(v['id'],df.id) and np.array_equal(v['target'],y) and np.array_equal(v['fold'],fold) for v in [z,nn,tree]);pred=np.full(len(df),np.nan)
        for k in range(K):
            fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);order=np.random.default_rng(20261031+k+(0 if stage=='fivefold' else 10000)).permutation(fi);allcontexts=[];p=np.zeros(len(vi))
            for j in range(4):
                run=out/f'fold{k}/context{j}';r=json.loads((run/'result.json').read_text());idx=np.load(run/'context.npz');v=np.load(run/'validation.npz');context=order[j*32768:(j+1)*32768]
                assert np.array_equal(idx['idx'],context) and np.array_equal(idx['id'],df.id.iloc[context]) and np.array_equal(v['idx'],vi) and np.array_equal(v['id'],df.id.iloc[vi]) and np.array_equal(v['target'],y[vi]) and not np.intersect1d(context,vi).size
                assert r['status']=='complete' and r['context_rows']==32768 and r['train_rows']==len(fi) and r['valid_rows']==len(vi) and r['feature_width']==48 and r['norm_method']==plan['norm_methods_per_context'][j] and r['random_state']==42+j
                assert r['runner_sha256']==plan['source_sha256']['research_r36_worker.py'] and r['plan_sha256']==sha(ROOT/'research_r36_plan.json') and r['context_sha256']==sha(run/'context.npz') and r['prediction_sha256']==sha(run/'validation.npz')
                assert not r['outer_validation_used_for_fit_or_preprocessing'] and not r['pretrained_weight_update'] and r['cache_devices']==['cpu'] and r['predicted_rows']==len(vi) and r['repeat_error']<1e-6 and r['peak_gpu_process_bytes']<=1073741824 and r['peak_allocated_bytes']<=778043392 and r['peak_rss_bytes']<=8589934592
                assert r['counters']['preallocated_ff_calls']>0 and r['counters']['staged_cache_entries']>0
                assert np.isfinite(v['pred']).all() and ((v['pred']>=0)&(v['pred']<=1)).all();p+=v['pred']/4;allcontexts.extend(context.tolist());records.append(dict(stage=stage,fold=k,context=j,context_sha256=r['context_sha256'],prediction_sha256=r['prediction_sha256'],norm_method=r['norm_method']))
            assert len(allcontexts)==len(set(allcontexts))==131072;pred[vi]=p
        assert np.max(np.abs(pred-z['pred']))<1e-12;reference=pool(tree['tree'],nn['pred']);assert np.max(np.abs(reference-nn['candidate']))<1e-12;candidate=combine(reference,pred)
        assert np.max(np.abs(reference-z['reference_v11']))<1e-12 and np.max(np.abs(candidate-z['candidate']))<1e-12;checks[stage]=evidence(y,reference,candidate,fold);assert checks[stage]['eligible_for_new_partition_replication']
    r=dict(status='metadata_passed_pending_all_context_replays',audited_utc=now(),plan_sha256=sha(ROOT/'research_r36_plan.json'),verified_context_records=len(records),context_records=records,validation=checks,disjoint_context_membership_verified=True,positive_class='True=1',fixed_schedule_old_new_pipeline_parity_error=0,source_auxiliary_models_reused=65,all_GPU_replays_required_before_test=True,limitation='ReusedofficialSATlabels;conditionalboundsomitadaptivehistory;no untouchedholdoutclaim.')
    dump(ROOT/'research_r36_metadata_audit.json',r);print('METADATA_PASS',len(records),flush=True)
if __name__=='__main__':main()

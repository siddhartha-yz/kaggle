"""Pin all contexts/results and source inputs after both validation gates pass."""
from research_r4_common import ROOT,now,dump,sha
import json,importlib.metadata

def main():
    assert not (ROOT/'research_r36_final_inference_plan.json').exists()
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r36_tabicl48_{stage}/result.json').read_text())['eligible']
    inherited=dict(json.loads((ROOT/'research_r23_final_inference_plan.json').read_text())['inherited_artifact_sha256'])
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    def pin(n):inherited[n]=sha(ROOT/n)
    for n in ['research_r36_plan.json','research_r23_final_inference_plan.json','research_r23_pre_inference_audit.json','research_r23_delivery_audit.json','submission_conditional_neural_r23_v11.csv','research_artifacts/r23_conditional_nn_full/test_components.npz','research_artifacts/r23_conditional_nn_full/result.json','research_artifacts/r36_memory_parity/result.json','research_artifacts/r36_tabicl48_probe_32768/result.json']:pin(n)
    sr=json.loads((ROOT/'research_artifacts/r20_aux_officialcb/result.json').read_text())
    for row in sr['models']:
        n=f"research_artifacts/r20_aux_officialcb/rating{row['rating_index']}_fold{row['auxiliary_fold']}/model.cbm";assert sha(ROOT/n)==row['model_sha256'];pin(n)
    for stage,K in [('fivefold',5),('threefold',3)]:
        stem=f'research_artifacts/r36_tabicl48_{stage}'
        for n in ['result.json','oof.npz']:pin(f'{stem}/{n}')
        for k in range(K):
            for j in range(4):
                for n in ['result.json','context.npz','validation.npz']:pin(f'{stem}/fold{k}/context{j}/{n}')
    names=['research_r36_audit.py','research_r36_replay.py','research_r36_test_worker.py','research_r36_csv.py','research_r36_freeze.py','research_r36_post_validation.py','research_r36_worker.py','research_r36_check.py','research_r36_memory.py','research_r34_features.py','research_r22_tabicl_offload.py','research_w3_tabicl_model.py']
    code={n:sha(ROOT/n) for n in names}
    for n in ['layers','attention','encoders','learning','tabicl','inference','kv_cache']:
        rel=f'../.venv-tabicl-w3/lib/python3.12/site-packages/tabicl/_model/{n}.py';code[rel]=sha(ROOT/rel)
    rel='../.venv-tabicl-w3/lib/python3.12/site-packages/tabicl/_sklearn/classifier.py';code[rel]=sha(ROOT/rel)
    r=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r36_plan.json'),code_sha256=code,inherited_artifact_sha256=inherited,all32_context_reconstruction_required_before_test=True,large32768_pointwise_FFN_bitwise_proof_required=True,final_prediction='uniformmean20savedfivefoldcontexts, .125logitshare, originalv11.875logitshare;clip1e-6',new_pretrained_weight_update=False,parent_only_submitter=True,remaining_daily_UI_quota_observed=6,quota_recheck_required=True,actual_top500_verified=False)
    dump(ROOT/'research_r36_final_inference_plan.json',r);print('FROZEN',len(inherited),flush=True)
if __name__=='__main__':main()

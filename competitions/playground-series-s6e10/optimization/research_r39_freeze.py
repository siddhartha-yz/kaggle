"""Pin audit/inference inputs only after both fixed validation stages qualify."""
from research_r4_common import ROOT,now,dump,sha
import json

def main():
    assert not (ROOT/'research_r39_final_inference_plan.json').exists()
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r39_uniform_{stage}/result.json').read_text())['eligible']
    inherited=dict(json.loads((ROOT/'research_r23_final_inference_plan.json').read_text())['inherited_artifact_sha256'])
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    def pin(n):inherited[n]=sha(ROOT/n)
    for n in ['research_r39_plan.json','research_r39_gate_alias.json','research_r23_final_inference_plan.json','research_r23_pre_inference_audit.json','research_r23_delivery_audit.json','submission_conditional_neural_r23_v11.csv','research_artifacts/r23_conditional_nn_full/test_components.npz','research_artifacts/r23_conditional_nn_full/result.json']:pin(n)
    sr=json.loads((ROOT/'research_artifacts/r20_aux_officialcb/result.json').read_text())
    for row in sr['models']:
        n=f"research_artifacts/r20_aux_officialcb/rating{row['rating_index']}_fold{row['auxiliary_fold']}/model.cbm";assert sha(ROOT/n)==row['model_sha256'];pin(n)
    for stage,K in [('fivefold',5),('threefold',3)]:
        for n in ['result.json','oof.npz','completed.json']:pin(f'research_artifacts/r39_uniform_{stage}/{n}')
        for seed in [43,44,45]:
            for k in range(K):
                for n in ['result.json','model.pkl','transformer.joblib','validation.npz']:pin(f'research_artifacts/r39_seed{seed}_lean_{stage}/fold{k}/{n}')
    code={n:sha(ROOT/n) for n in ['research_r39_audit.py','research_r39_csv.py','research_r39_freeze.py','research_r39_loader.py','research_r39_worker.py','research_r39_assemble.py','research_r39_check.py']}
    result=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r39_plan.json'),code_sha256=code,inherited_artifact_sha256=inherited,new_training=False,final_formula='Unchangedtree withNN640uniform4blocks42(existing160),43(160),44(160),45(160), equal clippedlogpool',test_models='15new savedfivefoldmodels32members each+cachedseed42testmean160',all_24_new_models_CPU_replay_required=True,old_seed42_not_retrained=True,parent_only_submitter=True,remaining_daily_UI_quota_observed=6,quota_recheck_required=True,actual_top500_verified=False)
    dump(ROOT/'research_r39_final_inference_plan.json',result);print('FROZEN',len(inherited),flush=True)
if __name__=='__main__':main()

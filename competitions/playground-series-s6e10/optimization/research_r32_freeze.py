"""Freeze complete inference/audit inputs only after both validation gates pass."""
from research_r4_common import ROOT,now,dump,sha
import json

def main():
    assert not (ROOT/'research_r32_final_inference_plan.json').exists()
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r32_tabm74_{stage}/result.json').read_text())['eligible']
    prior=json.loads((ROOT/'research_r23_final_inference_plan.json').read_text())
    inherited=dict(prior['inherited_artifact_sha256'])
    # The prior audit was pinned by v11; preserve its complete dependency hashes.
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    def pin(n):inherited[n]=sha(ROOT/n)
    for n in ['research_r23_final_inference_plan.json','research_r23_pre_inference_audit.json','research_r23_delivery_audit.json','submission_conditional_neural_r23_v11.csv','research_artifacts/r23_conditional_nn_full/test_components.npz','research_artifacts/r23_conditional_nn_full/result.json','research_r32_plan.json','research_artifacts/r32_conditional_tabm_hardware/result.json']:pin(n)
    sr=json.loads((ROOT/'research_artifacts/r20_aux_officialcb/result.json').read_text())
    for r in sr['models']:
        n=f"research_artifacts/r20_aux_officialcb/rating{r['rating_index']}_fold{r['auxiliary_fold']}/model.cbm";assert sha(ROOT/n)==r['model_sha256'];pin(n)
    for stage,K in [('fivefold',5),('threefold',3)]:
        stem=f'research_artifacts/r32_tabm74_{stage}'
        for n in ['result.json','oof.npz']:pin(f'{stem}/{n}')
        for k in range(K):
            for n in ['result.json','inner_split.npz','selection.json','selection.pt','selection_transformer.joblib','model.json','model.pt','transformer.joblib','validation.npz']:pin(f'{stem}/fold{k}/{n}')
    code={n:sha(ROOT/n) for n in ['research_r32_audit.py','research_r32_csv.py','research_r32_freeze.py','research_r32_features.py','research_r32_engine.py','research_r32_check.py']}
    result=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r32_plan.json'),code_sha256=code,inherited_artifact_sha256=inherited,final_formula='sigmoid(.5logit(currenttree)+.375logit(NN74mean160)+.125logit(TabM74mean80)); clipping1e-6',new_training=False,newtestmodels='five saved outer TabM16models, uniformmean80members; three confirmationmodels excluded',validation_limitation='Historicalvalidationlabelreuse;conditionalboundsomitadaptivehistory',parent_only_upload=True,quota_recheck_required=True,remaining_daily_UI_quota_observed=6,actual_top500_verified=False)
    dump(ROOT/'research_r32_final_inference_plan.json',result);print('FROZEN',len(inherited),flush=True)
if __name__=='__main__':main()

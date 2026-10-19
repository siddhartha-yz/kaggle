"""Seal five saved gate models and exact actual-v11 test experts; no new fitting."""
from research_r4_common import ROOT,now,dump,sha
import json
def main():
    audit=json.loads((ROOT/'research_r57_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_gate_models']==8 and audit['verified_inner_NN_models']==8 and audit['verified_inner_tree_models']==32 and audit['main_plan_sha256']==sha(ROOT/'research_r57_plan.json')
    names=['research_r57_pre_inference_audit.json','research_r57_audit_plan.json','research_r57_audit.py','research_r57_csv.py','research_r57_prepare_final.py','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','submission_conditional_neural_r23_v11.csv']
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    for stage,K in [('fivefold',5),('threefold',3)]:
        path=f'research_artifacts/r57_nested_gate_{stage}/result.json';assert json.loads((ROOT/path).read_text())['eligible'];names.extend([path,f'research_artifacts/r57_nested_gate_{stage}/oof.npz'])
        for k in range(K):names.extend([f'research_artifacts/r57_nested_gate_{stage}/fold{k}/{n}' for n in ['result.json','gate.joblib']])
    result=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r57_plan.json'),input_sha256={n:sha(ROOT/n) for n in names},final_gate_count=5,coefficient_average='Uniform arithmetic mean across five saved outer gates, absentcohort coefficient0; apply once to clipped currenttree+NN160 logodds',new_model_fits=0,test_GPU_used=False,parent_only_submitter=True)
    path=ROOT/'research_r57_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());result['frozen_utc']=old['frozen_utc'];assert result==old
    else:dump(path,result)
    print('R57_FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

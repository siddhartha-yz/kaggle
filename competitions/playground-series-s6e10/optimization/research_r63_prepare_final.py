"""Seal the saved160-member final inference after independent eight-model audit."""
from research_r4_common import ROOT,now,dump,sha
import json
def main():
    a=json.loads((ROOT/'research_r63_pre_inference_audit.json').read_text());assert a['status']=='passed' and a['verified_model_records']==8 and a['verified_native_neural_members']==256 and a['plan_sha256']==sha(ROOT/'research_r63_plan.json')
    code=['research_r63_csv.py','research_r63_audit.py','research_r63_prepare_final.py','research_r63_loader.py','research_r23_guard.py']
    names=['research_r63_pre_inference_audit.json','research_r63_audit_plan.json','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','submission_conditional_neural_r23_v11.csv']
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    for stage,K in [('fivefold',5),('threefold',3)]:
        result=f'research_artifacts/r63_cohortclass_lean_{stage}/result.json';assert json.loads((ROOT/result).read_text())['eligible'];names.extend([result,f'research_artifacts/r63_cohortclass_lean_{stage}/oof.npz'])
        for k in range(K):names.extend([f'research_artifacts/r63_cohortclass_lean_{stage}/fold{k}/{n}' for n in ['result.json','model.pkl','transformer.joblib']])
    r=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r63_plan.json'),code_sha256={n:sha(ROOT/n) for n in code},inherited_artifact_sha256={n:sha(ROOT/n) for n in names},no_new_model_fitting=True,native_test_members=160,parent_only_submitter=True,quota_recheck_before_upload=True)
    path=ROOT/'research_r63_final_inference_plan.json'
    if path.exists():
        prior=json.loads(path.read_text());r['frozen_utc']=prior['frozen_utc'];assert r==prior
    else:dump(path,r)
    print('R56_FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

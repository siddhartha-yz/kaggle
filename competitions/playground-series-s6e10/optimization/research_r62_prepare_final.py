"""Freeze one fullCPU LGB97 fit and exact formal-v11 test components after both gates."""
from research_r4_common import ROOT,now,dump,sha
import json
def main():
    audit=json.loads((ROOT/'research_r62_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_student_models']==8 and audit['verified_source_class_models']==13 and audit['main_plan_sha256']==sha(ROOT/'research_r62_plan.json')
    names=['research_r62_pre_inference_audit.json','research_r62_audit_plan.json','research_r62_audit.py','research_r62_csv.py','research_r62_prepare_final.py','research_artifacts/original_teacher_v1/test_features.npz','research_artifacts/r21_probability_lgb_officialcb_full/transformer.joblib','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','submission_conditional_neural_r23_v11.csv']
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    for stage in ['fivefold','threefold']:
        n=f'research_artifacts/r62_cohortclass_lgb97_{stage}/result.json';assert json.loads((ROOT/n).read_text())['eligible'];names.extend([n,f'research_artifacts/r62_cohortclass_lgb97_{stage}/oof.npz'])
    r=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r62_plan.json'),csv_source_sha256=sha(ROOT/'research_r62_csv.py'),input_sha256={n:sha(ROOT/n) for n in names},full_model='Same639LGB97CPU6fitall699635',formula='Replaceonly.46875LGBshare;preserveactualv11other3treefamiliesandNN160;equalclippedlogitpool',parent_only_submitter=True);path=ROOT/'research_r62_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());r['frozen_utc']=old['frozen_utc'];assert r==old
    else:dump(path,r)
    print('R60_FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

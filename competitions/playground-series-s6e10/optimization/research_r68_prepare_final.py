"""Freeze saved-model-only test inference after both gates and all eight independent audits."""
from research_r4_common import ROOT,now,dump,sha
import json

def main():
    a=json.loads((ROOT/'research_r68_pre_inference_audit.json').read_text());assert a['status']=='passed' and a['verified_model_records']==8 and a['verified_native_neural_members']==256 and a['plan_sha256']==sha(ROOT/'research_r68_plan.json')
    names=['research_r68_pre_inference_audit.json','research_r68_audit_plan.json','research_r68_audit.py','research_r68_csv.py','research_r68_prepare_final.py','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','submission_conditional_neural_r23_v11.csv'];assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf';assert sha(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz')=='dfbde5f8224975da897e6b12d9ed1f837c786c70eaad3e0c48f3934e101af0d6'
    for stage,K in [('fivefold',5),('threefold',3)]:
        path=f'research_artifacts/r68_pure_ranknet_{stage}/result.json';assert json.loads((ROOT/path).read_text())['eligible'];names.extend([path,f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz'])
        for k in range(K):names.extend([f'research_artifacts/r68_pure_ranknet_{stage}/fold{k}/{n}' for n in ['result.json','model.pkl','transformer.joblib','validation.npz','calibration.json','calibration_training.npz']])
    result=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r68_plan.json'),input_sha256={n:sha(ROOT/n) for n in names},new_classifier_fits=0,native_new_test_members=160,parent_only_submitter=True,quota_recheck_before_upload=True)
    path=ROOT/'research_r68_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());result['frozen_utc']=old['frozen_utc'];assert result==old
    else:dump(path,result)
    print('FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

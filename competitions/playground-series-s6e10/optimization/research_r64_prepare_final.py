"""Freeze qualified audited 960-member inference; no new classifier fit."""
from research_r4_common import ROOT,now,dump,sha
import json

def main():
    plan=json.loads((ROOT/'research_r64_plan.json').read_text());audit=json.loads((ROOT/'research_r64_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_model_records']==48 and audit['verified_native_neural_members']==1536 and audit['plan_sha256']==sha(ROOT/'research_r64_plan.json')
    names=['research_r64_pre_inference_audit.json','research_r64_audit_plan.json','research_r64_rank_audit.json','research_r64_execution_plan.json','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','submission_conditional_neural_r23_v11.csv']
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    for stage,K in [('fivefold',5),('threefold',3)]:
        n=f'research_artifacts/r64_viewbag_{stage}/result.json';assert json.loads((ROOT/n).read_text())['eligible'];names.extend([n,f'research_artifacts/r64_viewbag_{stage}/oof.npz'])
        for member in plan['members']:
            directory=f"{member['directory']}_{stage}" if stage=='fivefold' or member['name']=='r23' else f"r64_{member['name']}_{stage}"
            for k in range(K):names.extend([f'research_artifacts/{directory}/fold{k}/{n}' for n in ('result.json','model.pkl','transformer.joblib','validation.npz')])
    code=['research_r64_csv.py','research_r64_audit.py','research_r64_prepare_final.py','research_r23_guard.py',*plan['source_sha256']]
    r=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r64_plan.json'),code_sha256={n:sha(ROOT/n) for n in code},inherited_artifact_sha256={n:sha(ROOT/n) for n in names},no_new_classifier_fitting=True,native_test_members=960,parent_only_submitter=True,quota_recheck_before_upload=True)
    path=ROOT/'research_r64_final_inference_plan.json'
    if path.exists():
        prior=json.loads(path.read_text());r['frozen_utc']=prior['frozen_utc'];assert r==prior
    else:dump(path,r)
    print('FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

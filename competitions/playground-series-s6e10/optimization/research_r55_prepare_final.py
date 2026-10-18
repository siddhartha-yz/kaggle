"""Freeze final inputs only after both gates and the independent eight-model audit."""
from research_r4_common import ROOT,now,dump,sha
import json
def main():
    audit=json.loads((ROOT/'research_r55_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_models']==8 and audit['main_plan_sha256']==sha(ROOT/'research_r55_plan.json')
    names=['research_r55_pre_inference_audit.json','research_r55_audit_plan.json','research_r55_audit.py','research_r55_csv.py','research_r55_prepare_final.py','submission_conditional_neural_r23_v11.csv','research_artifacts/original_teacher_v1/result.json','research_artifacts/original_teacher_v1/test_features.npz','research_artifacts/r21_probability_lgb_officialcb_full/transformer.joblib']
    for stage in ['fivefold','threefold']:
        r=ROOT/f'research_artifacts/r55_randomized_lgb84_{stage}/result.json';assert json.loads(r.read_text())['eligible'];names.extend([str(r.relative_to(ROOT)),f'research_artifacts/r55_randomized_lgb84_{stage}/oof.npz'])
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    frozen=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r55_plan.json'),input_sha256={n:sha(ROOT/n) for n in names},full_data_selection='Same inner80/20 stratified seed20261025; AUC ES200 max12000; fresh refit exact selected count',formula='.875 logit exact submitted v11 +.125 logit randomized LGB84, clipped1e-6',parent_only_submitter=True)
    path=ROOT/'research_r55_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());frozen['frozen_utc']=old['frozen_utc'];assert old==frozen
    else:dump(path,frozen)
    print('FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

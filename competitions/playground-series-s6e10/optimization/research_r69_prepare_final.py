"""Freeze final CPU-only gate inference after both gates and independent nested audits."""
from research_r4_common import ROOT,now,dump,sha
import json

def main():
    a=json.loads((ROOT/'research_r69_pre_inference_audit.json').read_text());assert a['status']=='passed' and [a[k] for k in ['verified_inner_tree_models','verified_inner_NNCE_models','verified_inner_RankNet_models','verified_scalar_gates']]==[32,8,8,8] and a['main_plan_sha256']==sha(ROOT/'research_r69_plan.json')
    names=['research_r69_pre_inference_audit.json','research_r69_audit_plan.json','research_r69_audit.py','research_r69_csv.py','research_r69_prepare_final.py','research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz','research_artifacts/r23_conditional_nn_full/test_components.npz','research_artifacts/r68_pure_ranknet_full/test_components.npz','submission_pure_ranknet_r68_v12.csv','research_r68_delivery_audit.json','research_r68_final_inference_plan.json']
    prior=json.loads((ROOT/'research_r68_final_inference_plan.json').read_text())
    for n,h in prior['input_sha256'].items():assert sha(ROOT/n)==h,n
    a68=json.loads((ROOT/'research_r68_delivery_audit.json').read_text());assert sha(ROOT/'submission_pure_ranknet_r68_v12.csv')==a68['sha256']=='f48276b144e8c9297fb9fec27174044a55a7eb3f2af23dde3f13478a38a4e545' and sha(ROOT/'research_artifacts/r68_pure_ranknet_full/test_components.npz')==a68['test_components_sha256']
    for stage,K in [('fivefold',5),('threefold',3)]:
        path=f'research_artifacts/r69_rank_gate_{stage}/result.json';assert json.loads((ROOT/path).read_text())['eligible'];names.extend([path,f'research_artifacts/r69_rank_gate_{stage}/oof.npz'])
        for k in range(K):names.extend([f'research_artifacts/r69_rank_gate_{stage}/fold{k}/{n}' for n in ['result.json','gate.joblib','validation.npz','inner_indices.npz']])
    result=dict(frozen_utc=now(),main_plan_sha256=sha(ROOT/'research_r69_plan.json'),input_sha256={n:sha(ROOT/n) for n in names},new_classifier_fits=0,new_GPU_model_predictions=0,parent_only_submitter=True,quota_recheck_before_upload=True)
    path=ROOT/'research_r69_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());result['frozen_utc']=old['frozen_utc'];assert result==old
    else:dump(path,result)
    print('FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

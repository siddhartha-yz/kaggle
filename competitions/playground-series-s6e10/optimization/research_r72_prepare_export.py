"""Freeze CSV serialization and independent verification afterall20 predictions exist."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    t=json.loads((ROOT/'research_r72_test_complete.json').read_text());assert t['status']=='complete' and t['total_contexts']==20 and t['prediction_plan_sha256']==sha(ROOT/'research_r72_final_inference_plan.json');assert json.loads((ROOT/'research_r72_pre_inference_audit.json').read_text())['status']=='passed'
    plan=json.loads((ROOT/'research_r72_plan.json').read_text());assert 'fixed.875actualv12/.125TabICL' in plan['blend'] and 'mean20contextprobs' in plan['audit_and_final']
    names=['research_r72_plan.json','research_r72_final_inference_plan.json','research_r72_test_complete.json','research_r72_pre_inference_audit.json','research_r72_csv.py','research_r72_verify_csv.py','research_r72_prepare_export.py','data/test.csv','data/sample_submission.csv','submission_pure_ranknet_r68_v12.csv']
    for k in range(5):
        for j in range(4):
            run=f'research_artifacts/r72_largecontext_test/fold{k}_context{j}';r=json.loads((ROOT/run/'result.json').read_text());assert r['status']=='complete' and r['rows']==299844 and r['test_prediction_sha256']==sha(ROOT/run/'test_predictions.npz');names.extend([run+'/result.json',run+'/test_predictions.npz'])
    result=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r72_plan.json'),prediction_plan_sha256=sha(ROOT/'research_r72_final_inference_plan.json'),input_sha256={n:sha(ROOT/n) for n in names},reference_weight=.875,TabICL_weight=.125,probability_clip=1e-6,context_count=20,context_mean='uniform arithmetic probability mean',CSV_rows=299844,positive_class='True=1',new_classifier_weight_fits=0,parent_only_submitter=True,quota_recheck_before_upload=True)
    p=ROOT/'research_r72_export_plan.json'
    if p.exists():
        old=json.loads(p.read_text());result['frozen_utc']=old['frozen_utc'];assert old==result
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    print('EXPORT_PLAN_FROZEN',sha(p),flush=True)
if __name__=='__main__':main()

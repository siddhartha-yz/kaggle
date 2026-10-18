"""Freeze final inference inputs only after both gates and the independent32-context audit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((ROOT/'research_r54_plan.json').read_text());auditplan=json.loads((ROOT/'research_r54_audit_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**auditplan['code_sha256'],**auditplan['input_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r54_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_contexts']==32 and audit['plan_sha256']==sha(ROOT/'research_r54_plan.json')
    names=['data/test.csv','data/sample_submission.csv','submission_conditional_neural_r23_v11.csv','research_r54_pre_inference_audit.json','research_r54_static_audit.json','research_r54_pretrained_provenance.json','research_r54_audit_plan.json']
    assert sha(ROOT/'submission_conditional_neural_r23_v11.csv')=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf'
    for stage in ['fivefold','threefold']:
        result=json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/result.json').read_text());assert result['eligible'];names.extend([f'research_artifacts/r54_tabpfn_v2_{stage}/result.json',f'research_artifacts/r54_tabpfn_v2_{stage}/oof.npz'])
    for k in range(5):
        for m in range(4):
            prefix=f'research_artifacts/r54_tabpfn_v2_fivefold/fold{k}/context{m}';names.extend(prefix+'/'+n for n in ['result.json','context.npz','transformer.joblib','validation.npz','independent_GPU_audit.json'])
    result=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r54_plan.json'),code_sha256=auditplan['code_sha256'],input_sha256={n:sha(ROOT/n) for n in names},query_batch_rows=512,model_contexts=20,formula=plan['blend'],baseline='Actual byte-verified officially submitted v11 CSV',attribution='Built with PriorLabs-TabPFN',parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,statistics_limitation=plan['statistics_limitation'])
    path=ROOT/'research_r54_final_inference_plan.json'
    if path.exists():
        old=json.loads(path.read_text());assert all(old[k]==v for k,v in result.items() if k!='frozen_utc')
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL_PLAN_FROZEN',sha(path),flush=True)
if __name__=='__main__':main()

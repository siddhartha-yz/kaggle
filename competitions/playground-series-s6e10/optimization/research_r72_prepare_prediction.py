"""Freeze saved-context test inference independently of later CSV serialization."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    a=json.loads((ROOT/'research_r72_pre_inference_audit.json').read_text());assert a['status']=='passed' and a['verified_contexts']==32 and a['main_plan_sha256']==sha(ROOT/'research_r72_plan.json')
    names=['research_r72_pre_inference_audit.json','research_r72_audit_plan.json','research_r72_audit_worker.py','research_r72_test_worker.py','research_r72_test_driver.py','research_r72_prepare_prediction.py','research_r72_headroom.py','research_r72_package_integrity.json','research_sources/tabicl/tabicl-classifier-v2-20260212.ckpt','data/train.csv','data/test.csv','data/sample_submission.csv','submission_pure_ranknet_r68_v12.csv'];assert sha(ROOT/'submission_pure_ranknet_r68_v12.csv')=='f48276b144e8c9297fb9fec27174044a55a7eb3f2af23dde3f13478a38a4e545'
    for stage,K in [('fivefold',5),('threefold',3)]:
        out=f'research_artifacts/r72_largecontext_raw21_{stage}';assert json.loads((ROOT/out/'result.json').read_text())['eligible'];names.extend([out+'/result.json',out+'/oof.npz'])
        for k in range(K):
            for j in range(4):
                names.extend([f'{out}/fold{k}/context{j}/{n}' for n in ['result.json','context.npz','validation.npz']]);names.append(f'research_artifacts/r72_independent_audit/{stage}_fold{k}_context{j}/result.json')
    result=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r72_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r72_pre_inference_audit.json'),input_sha256={n:sha(ROOT/n) for n in names},model_contexts=20,new_classifier_weight_fits=0,parent_only_submitter=True,quota_recheck_before_upload=True,serializer_scope='CSVexport gets separate freeze afterall20testpredictions; no modelreplay needed for a serialization-only repair.')
    p=ROOT/'research_r72_final_inference_plan.json'
    if p.exists():
        old=json.loads(p.read_text());result['frozen_utc']=old['frozen_utc'];assert result==old
    else:p.write_text(json.dumps(result,indent=2)+'\n')
    print('PREDICTION_PLAN_FROZEN',sha(p),flush=True)
if __name__=='__main__':main()

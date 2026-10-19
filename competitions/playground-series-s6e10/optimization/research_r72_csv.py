"""Serialize the one frozen larger-context prediction mixture after all audits."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,resource
import numpy as np,pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    export=json.loads((ROOT/'research_r72_export_plan.json').read_text());audit=json.loads((ROOT/'research_r72_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_contexts']==32 and export['context_count']==20 and export['reference_weight']==.875 and export['TabICL_weight']==.125 and export['probability_clip']==1e-6
    for n,h in export['input_sha256'].items():assert sha(ROOT/n)==h,n
    test=pd.read_csv(ROOT/'data/test.csv',usecols=['id']);sample=pd.read_csv(ROOT/'data/sample_submission.csv');old=pd.read_csv(ROOT/'submission_pure_ranknet_r68_v12.csv');assert sha(ROOT/'submission_pure_ranknet_r68_v12.csv')=='f48276b144e8c9297fb9fec27174044a55a7eb3f2af23dde3f13478a38a4e545' and len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and np.array_equal(test.id,old.id)
    total=np.zeros(len(test));records=[]
    for k in range(5):
        for j in range(4):
            run=ROOT/f'research_artifacts/r72_largecontext_test/fold{k}_context{j}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['prediction_plan_sha256']==sha(ROOT/'research_r72_final_inference_plan.json') and r['test_prediction_sha256']==sha(run/'test_predictions.npz');z=np.load(run/'test_predictions.npz');assert np.array_equal(z['id'],test.id) and np.isfinite(z['pred']).all() and ((z['pred']>=0)&(z['pred']<=1)).all();total+=z['pred']/20;z.close();records.append(r)
    new=np.clip(total,1e-6,1-1e-6);reference=np.clip(old.satisfaction.to_numpy(),1e-6,1-1e-6);candidate=expit(.875*logit(reference)+.125*logit(new));out=ROOT/'research_artifacts/r72_largecontext_full';out.mkdir(exist_ok=True);np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v12=old.satisfaction.to_numpy(),TabICL_mean20=total,candidate=candidate);csv=ROOT/'submission_largecontext_tabicl_r72_v13.csv'
    if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
    check=pd.read_csv(csv);err=float(np.max(np.abs(check.satisfaction-candidate)));assert list(check)==list(sample) and len(check)==299844 and check.id.is_unique and np.array_equal(check.id,test.id) and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all() and err<1e-14
    result=dict(status='pending_independent_CSV_verification',built_utc=datetime.now(timezone.utc).isoformat(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class='True=1',CSV_roundtrip_error=err,export_plan_sha256=sha(ROOT/'research_r72_export_plan.json'),prediction_plan_sha256=sha(ROOT/'research_r72_final_inference_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r72_pre_inference_audit.json'),test_components_sha256=sha(out/'test_components.npz'),serializer_sha256=sha(__file__),context_count=20,new_classifier_weight_fits=0,peak_GPU_bytes=max(r['peak_gpu_process_bytes'] for r in records),peak_model_RSS_bytes=max(r['peak_RSS_bytes'] for r in records),export_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,validation=audit['validation'],new_formal_submissions_since_v12=0,current_daily_quota=None,parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False);(ROOT/'research_r72_csv_build.json').write_text(json.dumps(result,indent=2)+'\n');print('CSV_BUILT_PENDING_INDEPENDENT_CHECK',csv.name,result['sha256'],flush=True)
if __name__=='__main__':main()

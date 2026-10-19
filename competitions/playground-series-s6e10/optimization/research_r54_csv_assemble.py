"""Assemble20 context probabilities, mix actualv11, validate parent-only CSV."""
from research_r4_common import ROOT,now,sha,dump
import json,numpy as np,pandas as pd
from scipy.special import expit,logit
def main():
    final=json.loads((ROOT/'research_r54_final_inference_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r54_plan.json')
    for n,h in {**final['code_sha256'],**final['input_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r54_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_contexts']==32
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');test=pd.read_csv(ROOT/'data/test.csv',usecols=['id']);basepath=ROOT/'submission_conditional_neural_r23_v11.csv';assert sha(basepath)=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf';base=pd.read_csv(basepath);assert len(test)==len(base)==len(sample)==299844 and np.array_equal(test.id,sample.id) and np.array_equal(test.id,base.id) and test.id.is_unique
    total=np.zeros(len(test),'float64');records=[];out=ROOT/'research_artifacts/r54_tabpfn_v2_full'
    for k in range(5):
        for m in range(4):
            run=out/f'fold{k}/context{m}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['native_estimators']==1 and r['final_plan_sha256']==sha(ROOT/'research_r54_final_inference_plan.json') and r['test_sha256']==sha(run/'test.npz');z=np.load(run/'test.npz');assert np.array_equal(z['id'],test.id);total+=z['pred'];records.append(r)
    family=total/20;candidate=expit(.875*logit(np.clip(base.satisfaction.to_numpy(),1e-6,1-1e-6))+.125*logit(np.clip(family,1e-6,1-1e-6)));assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all() and np.std(candidate)>.01
    np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),reference_v11=base.satisfaction.to_numpy(),TabPFN_v2=family,candidate=candidate)
    csv=ROOT/'submission_TabPFN_v2_context_r54_v12.csv'
    if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
    check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id) and check.id.is_unique and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all();error=float(np.max(np.abs(check.satisfaction.to_numpy()-candidate)));assert error<1e-14
    result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=299844,positive_class=True,context_models=20,native_estimators=20,formula='expit(.875logit(clip(actualv11,1e-6,1-1e-6))+.125logit(clip(mean20TabPFNv2contextprobabilities,1e-6,1-1e-6)))',CSV_roundtrip_error=error,test_components_sha256=sha(out/'test_components.npz'),contexts=records,validation=audit['validation'],pre_inference_audit_sha256=sha(ROOT/'research_r54_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r54_plan.json'),final_plan_sha256=sha(ROOT/'research_r54_final_inference_plan.json'),attribution='Built with PriorLabs-TabPFN',additional_qualified_submissions_authorized=True,new_formal_submissions=0,parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,limitations='Reusedvalidationlabels,conditionalCIsomitadaptivehistory;4contextOOFversus20contexttestmean changesvariance. No leaderboard result established.');dump(out/'result.json',result);dump(ROOT/'research_r54_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
if __name__=='__main__':main()

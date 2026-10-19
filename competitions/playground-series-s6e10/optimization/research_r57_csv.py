"""Average five saved gate coefficients over exact v11 test experts; parent-only CSV."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,joblib,threading,resource,numpy as np,pandas as pd
from scipy.special import expit,logit
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r23_loader import pool
from research_r57_gate import logits
def main():
    frozen=json.loads((ROOT/'research_r57_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r57_plan.json').read_text());assert frozen['main_plan_sha256']==sha(ROOT/'research_r57_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**frozen['input_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r57_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_gate_models']==8
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r57_csv_resource_failure.json'),daemon=True);monitor.start()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and 'satisfaction' not in test;raw=test.drop(columns='id');t=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');n=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');old=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv');assert np.array_equal(t['id'],test.id) and np.array_equal(n['id'],test.id) and np.array_equal(old.id,test.id) and np.max(np.abs(t['newtree']-n['v10tree']))<1e-12 and np.max(np.abs(pool(t['newtree'],n['newNN'])-old.satisfaction))<1e-14
        coef=[];records=[]
        for k in range(5):
            run=ROOT/f'research_artifacts/r57_nested_gate_fivefold/fold{k}';r=json.loads((run/'result.json').read_text());assert r['gate_sha256']==sha(run/'gate.joblib');gate=joblib.load(run/'gate.joblib');assert gate.ridge==100. and (np.abs(gate.theta)<=.5).all();coef.append(gate.coefficients(raw));records.append(dict(fold=k,gate_sha256=r['gate_sha256']))
        mean=np.mean(coef,axis=0);a,d=logits(t['newtree'],n['newNN']);candidate=expit(a+mean*d);independent=np.mean([a+c*d for c in coef],axis=0);assert np.max(np.abs(candidate-expit(independent)))<1e-14 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();out=ROOT/'research_artifacts/r57_nested_gate_full';out.mkdir(exist_ok=True);np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v11tree=t['newtree'],v11NN=n['newNN'],mean_gate_theta=mean,candidate=candidate)
        csv=ROOT/'submission_nested_cohort_gate_r57_v12.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and check.id.is_unique and np.array_equal(check.id,test.id) and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all();roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert roundtrip<1e-14
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class='True=1',gate_records=records,final_gate_count=5,new_model_fits=0,test_GPU_used=False,CSV_roundtrip_error=roundtrip,main_plan_sha256=sha(ROOT/'research_r57_plan.json'),final_inference_plan_sha256=sha(ROOT/'research_r57_final_inference_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r57_pre_inference_audit.json'),baseline_CSV_sha256=sha(ROOT/'submission_conditional_neural_r23_v11.csv'),test_components_sha256=sha(out/'test_components.npz'),validation=audit['validation'],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CPU_threads=6,parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,formula=frozen['coefficient_average'],limitation=plan['statistics_limitation']);assert result['peak_RSS_bytes']<=8589934592;dump(out/'result.json',result);dump(ROOT/'research_r57_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

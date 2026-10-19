"""Saved scalar-gate CPU inference on unchanged, audited full v12 expert components."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,threading,resource,joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu

def main():
    plan=json.loads((ROOT/'research_r69_plan.json').read_text());final=json.loads((ROOT/'research_r69_final_inference_plan.json').read_text());audit=json.loads((ROOT/'research_r69_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_scalar_gates']==8 and final['main_plan_sha256']==sha(ROOT/'research_r69_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['input_sha256']}.items():assert sha(ROOT/n)==h,n
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r69_csv_resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');old=pd.read_csv(ROOT/'submission_pure_ranknet_r68_v12.csv');tree=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');nn=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');rank=np.load(ROOT/'research_artifacts/r68_pure_ranknet_full/test_components.npz')
        assert len(test)==299844 and test.id.is_unique and list(sample)==['id','satisfaction'] and np.array_equal(test.id,sample.id) and np.array_equal(test.id,old.id)
        for a in [tree,nn,rank]:assert np.array_equal(a['id'],test.id)
        ts,ns,rs=[np.clip(p.astype('float64'),1e-6,1-1e-6) for p in (tree['newtree'],nn['newNN'],rank['new_calibrated_NN'])];t,n,r=[logit(p) for p in (ts,ns,rs)];zero=expit(.875*.5*(t+n)+.125*r);assert np.max(np.abs(zero-old.satisfaction))<1e-14 and np.max(np.abs(rank['candidate']-old.satisfaction))<1e-14
        gates=[joblib.load(ROOT/f'research_artifacts/r69_rank_gate_fivefold/fold{k}/gate.joblib') for k in range(5)];theta=float(np.mean([g.theta for g in gates]));assert -.5<=theta<=.5 and all(g.ridge==100 and g.k==32 and g.seed==20261069 for g in gates)
        candidate=expit(.875*((.5-theta)*t+(.5+theta)*n)+.125*r);wt,wn,wr=.875*(.5-theta),.875*(.5+theta),.125;numerator=ts**wt*ns**wn*rs**wr;independent=numerator/(numerator+(1-ts)**wt*(1-ns)**wn*(1-rs)**wr);formula_error=float(np.max(np.abs(candidate-independent)));assert formula_error<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        out=ROOT/'research_artifacts/r69_rank_gate_full';out.mkdir(exist_ok=True);np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v12=old.satisfaction.to_numpy(),candidate=candidate);csv=ROOT/'submission_nested_pairgate_r69_v13.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert list(check)==list(sample) and len(check)==299844 and check.id.is_unique and np.array_equal(check.id,test.id) and roundtrip<1e-14 and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all()
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class='True=1',five_gate_coefficients=[g.theta for g in gates],mean_scalar_delta=theta,tree_weight=wt,NNCE_weight=wn,RankNet_weight=wr,new_classifier_fits=0,new_GPU_model_predictions=0,formula=plan['candidate_formula'],CSV_roundtrip_error=roundtrip,formula_error=formula_error,pre_inference_audit_sha256=sha(ROOT/'research_r69_pre_inference_audit.json'),main_plan_sha256=sha(ROOT/'research_r69_plan.json'),final_plan_sha256=sha(ROOT/'research_r69_final_inference_plan.json'),test_components_sha256=sha(out/'test_components.npz'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,validation=audit['validation'],new_formal_submissions_since_v12=0,current_daily_quota=None,quota_recheck_before_upload=True,parent_only_submitter=True,actual_top500_verified=False,limitation=plan['statistics_limitation']);assert result['peak_RSS_bytes']<=8589934592;dump(out/'result.json',result);dump(ROOT/'research_r69_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(ROOT/'research_r69_csv_failure.json',dict(error=repr(e),utc=now()));raise

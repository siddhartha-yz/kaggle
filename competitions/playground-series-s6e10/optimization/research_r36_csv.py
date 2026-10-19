"""CPU assembly of all20 audited context predictions into one validated CSV."""
from research_r4_common import ROOT,now,dump,sha
from research_r36_check import combine
from research_r15_loader import pool
import numpy as np,pandas as pd,json

def main():
    final=json.loads((ROOT/'research_r36_final_inference_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r36_plan.json')
    for n,h in {**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r36_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['reconstructed_contexts']==32
    test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id)
    previous=ROOT/'submission_conditional_neural_r23_v11.csv';assert sha(previous)=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf';ref=pd.read_csv(previous);assert np.array_equal(ref.id,test.id)
    old=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(old['id'],test.id);assert np.max(np.abs(pool(old['v10tree'],old['newNN'])-ref.satisfaction.to_numpy()))<1e-14
    p=np.zeros(len(test));records=[]
    for k in range(5):
        for j in range(4):
            src=ROOT/f'research_artifacts/r36_tabicl48_full/fold{k}/context{j}';r=json.loads((src/'result.json').read_text());assert r['status']=='complete' and r['rows']==len(test) and r['final_plan_sha256']==sha(ROOT/'research_r36_final_inference_plan.json') and r['prediction_sha256']==sha(src/'test.npz') and r['validation_reconstruction_error']<1e-6 and r['test_repeat_error']<1e-6 and r['peak_gpu_process_bytes']<=1073741824
            v=np.load(src/'test.npz');assert np.array_equal(v['id'],test.id) and np.isfinite(v['pred']).all();p+=v['pred']/20;records.append(r)
    candidate=combine(ref.satisfaction.to_numpy(),p);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();out=ROOT/'research_artifacts/r36_tabicl48_full';np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),TabICL48=p,reference_v11=ref.satisfaction.to_numpy(),candidate=candidate)
    destination=ROOT/'submission_conditional_tabicl_r36_v12.csv';assert not destination.exists();frame=pd.DataFrame(dict(id=test.id.to_numpy(),satisfaction=candidate));assert list(frame)==list(sample);frame.to_csv(destination,index=False,float_format='%.17g');re=pd.read_csv(destination);error=float(np.max(np.abs(re.satisfaction.to_numpy()-candidate)));assert error<1e-14 and np.array_equal(re.id,test.id) and len(re)==299844 and list(re)==['id','satisfaction'] and re.satisfaction.between(0,1).all()
    r=dict(status='complete',completed_utc=now(),csv=destination.name,rows=len(re),bytes=destination.stat().st_size,sha256=sha(destination),positive_class='True=1',uniform_context_models=20,context_rows_each=32768,fixed_formula='.875v11logit+.125logit(uniformmean20TabICL48contexts);clip1e-6',model_records=records,test_components_sha256=sha(out/'test_components.npz'),plan_sha256=sha(ROOT/'research_r36_plan.json'),final_plan_sha256=sha(ROOT/'research_r36_final_inference_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r36_pre_inference_audit.json'),CSV_roundtrip_error=error,IDs_schema_probabilities_verified=True,v11_exact_hash_and_formula_verified=True,peak_gpu_process_bytes=max(r['peak_gpu_process_bytes'] for r in records),new_formal_submissions=0,parent_only_submitter=True,remaining_daily_UI_quota_observed=6,quota_recheck_required=True,actual_top500_verified=False,limitation='HistoricalSATvalidationlabelreuse;conditionalboundsomitadaptivehistory.')
    dump(out/'result.json',r);dump(ROOT/'research_r36_delivery_audit.json',r);print('CSV_READY',r['csv'],r['sha256'],r['bytes'],flush=True)
if __name__=='__main__':main()

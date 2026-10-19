"""Source and class-map prerequisites before any source fit or official quality test."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,threading,numpy as np
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r62_source_data import data,observed_probability,CONDITION
def main():
    plan=json.loads((ROOT/'research_r62_source_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r62_source_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        raw,sat,train,test,cols,scope=data();assert set(np.unique(sat))=={0,1} and CONDITION not in raw and 'satisfaction' not in train and 'satisfaction' not in test
        class Fake:
            classes_=np.array([1,3,5])
            def predict_proba(self,query,**kwargs):return np.array([[.2,.3,.5],[.7,.2,.1],[.1,.6,.3],[.3,.3,.4]])
        actual,unknown=observed_probability(Fake(),np.zeros((4,1)),np.array([5,1,3,0]));assert np.array_equal(actual,[.5,.7,.6,0.]) and unknown==1
        result=dict(status='passed_original_SAT_only_no_official_SAT',audited_utc=now(),source_plan_sha256=sha(ROOT/'research_r62_source_plan.json'),scope=scope,source_positive_rows=int(sat.sum()),actual_class_position_and_unseen_class_zero_probability_test_passed=True,CPU_threads=6,peak_RSS_bytes=peak[0],quality_evaluated=False,new_model_fits=0);dump(ROOT/'research_r62_source_audit.json',result);print('R60_SOURCE_SCOPE_AND_CLASS_MAPPING_AUDIT_PASSED',flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

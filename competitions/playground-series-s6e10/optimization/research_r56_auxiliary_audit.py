"""Recheck unchanged cohort models and feature substitution without reading official SAT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,threading,resource
import pandas as pd,numpy as np
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r46_audit import audit_auxiliary
from research_r56_loader import auxiliary
from research_r23_loader import auxiliary as oldaux
from research_r23_features import ConditionalFeatures

def main():
    frozen=json.loads((ROOT/'research_r56_resource_plan.json').read_text())
    for n,h in {**frozen['source_sha256'],**frozen['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r55_driver_state.json').read_text())['status']=='not_qualified'
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r56_auxiliary_audit_resource_failure.json'),daemon=True);watcher.start()
    try:
        evidence,_=audit_auxiliary(json.loads((ROOT/'research_r46_plan.json').read_text()))
        columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0) if c!='satisfaction'];df=pd.read_csv(ROOT/'data/train.csv',usecols=columns);raw=df.drop(columns='id').iloc[:1024].reset_index(drop=True);y=np.arange(len(raw))%2;old=oldaux(df.id).iloc[:1024].reset_index(drop=True);new=auxiliary(df.id).iloc[:1024].reset_index(drop=True)
        a=ConditionalFeatures().fit_transform(raw,y,old);b=ConditionalFeatures().fit_transform(raw,y,new);pd.testing.assert_frame_equal(a.iloc[:,:47],b.iloc[:,:47],check_exact=True);pd.testing.assert_frame_equal(a.iloc[:,60:],b.iloc[:,60:],check_exact=True);assert b.shape==(1024,74) and b.select_dtypes(include='number').shape[1]==52 and np.array_equal(b.iloc[:,47:60].to_numpy(),new.iloc[:,:13].to_numpy('float32'))
        result=dict(status='passed_no_official_SAT_or_quality',audited_utc=now(),resource_plan_sha256=sha(ROOT/'research_r56_resource_plan.json'),cohort_auxiliary=evidence,semantic_proof=dict(rows=1024,labels='synthetic alternation only',official_SAT_values_parsed=False,feature_width=74,numeric_fields=52,nominal_fields=22,unchanged_fields_exact=61,replaced_fields_exact=13),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(ROOT/'research_r56_auxiliary_audit.json',result);print('R56_AUXILIARY_AND_VIEW_AUDIT_PASSED',flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

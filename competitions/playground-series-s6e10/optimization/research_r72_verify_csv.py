"""Independent full20-component/query-cache, geometric-formula and CSV-byte validation."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    e=json.loads((ROOT/'research_r72_export_plan.json').read_text());b=json.loads((ROOT/'research_r72_csv_build.json').read_text());assert b['export_plan_sha256']==sha(ROOT/'research_r72_export_plan.json') and e['reference_weight']==.875 and e['TabICL_weight']==.125 and e['context_count']==20 and e['probability_clip']==1e-6
    for n,h in e['input_sha256'].items():assert sha(ROOT/n)==h,n
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');testids=pd.read_csv(ROOT/'data/test.csv',usecols=['id']).id.to_numpy();old=pd.read_csv(ROOT/'submission_pure_ranknet_r68_v12.csv');csv=ROOT/b['csv'];q=pd.read_csv(csv);assert sha(csv)==b['sha256'] and csv.stat().st_size==b['bytes'] and list(q)==list(sample)==['id','satisfaction'] and len(q)==299844 and q.id.is_unique and np.array_equal(q.id,testids) and np.array_equal(q.id,sample.id) and np.array_equal(old.id,testids) and np.isfinite(q.satisfaction).all() and q.satisfaction.between(0,1).all();parts=[];chunks=0
    for k in range(5):
        for j in range(4):
            run=ROOT/f'research_artifacts/r72_largecontext_test/fold{k}_context{j}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['test_prediction_sha256']==sha(run/'test_predictions.npz') and r['prediction_plan_sha256']==e['prediction_plan_sha256'] and max(r['context_recreation_error'],r['test_A_B_A_error'],r['validation_replay_after_test_error'])<1e-6
            with np.load(run/'test_predictions.npz') as p:
                assert np.array_equal(p['id'],testids) and str(p['prediction_plan_sha256'].item())==e['prediction_plan_sha256'];v=p['pred'].copy()
            assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
            for start in range(0,len(testids),512):
                end=min(start+512,len(testids));idx=np.arange(start,end)
                with np.load(run/f'query{start:06d}.npz') as z:assert np.array_equal(z['idx'],idx) and np.array_equal(z['id'],testids[idx]) and str(z['prediction_plan_sha256'].item())==e['prediction_plan_sha256'] and np.array_equal(z['pred'],v[idx])
                chunks+=1
            parts.append(v)
    average=np.mean(np.stack(parts),axis=0);reference=np.clip(old.satisfaction.to_numpy(),1e-6,1-1e-6);new=np.clip(average,1e-6,1-1e-6);num=reference**.875*new**.125;independent=num/(num+(1-reference)**.875*(1-new)**.125);err=float(np.max(np.abs(independent-q.satisfaction)));assert err<1e-12
    cpath=ROOT/'research_artifacts/r72_largecontext_full/test_components.npz';assert b['test_components_sha256']==sha(cpath)
    with np.load(cpath) as c:assert np.array_equal(c['id'],testids) and np.max(np.abs(c['TabICL_mean20']-average))<1e-14 and np.max(np.abs(c['candidate']-independent))<1e-12
    report=dict(b,status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),independent_geometric_formula_error=err,independent_test_contexts_verified=20,independent_test_query_chunks_verified=chunks,verifier_sha256=sha(__file__),limitation='Repeatedhistoricalofficiallabels/recipes reused. Boundsconditionalomitadaptivehistory;3foldnotuntouchedholdout. OOF4contexts/test20changesvariance. Actualpublicranknotverified; parentonlyKaggleuploadafterfreshquota.');(ROOT/'research_r72_delivery_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('CHECKED_CSV_READY',report['csv'],report['sha256'],flush=True)
if __name__=='__main__':main()

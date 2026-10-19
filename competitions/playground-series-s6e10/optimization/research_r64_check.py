"""One fixed uniform ensemble over all six completed matched74 input views."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='6'
from research_r4_common import ROOT,sha,dump,now,evidence
from research_r15_loader import pool
import json,numpy as np,sys

def main(stage):
    plan=json.loads((ROOT/'research_r64_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    assert stage=='fivefold' or json.loads((ROOT/'research_artifacts/r64_viewbag_fivefold/result.json').read_text())['eligible']
    base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    predictions=[]
    for member in plan['members']:
        f=ROOT/'research_artifacts'/f"{member['directory']}_{stage}"/'oof.npz'
        if stage=='threefold' and member['name']!='r23':f=ROOT/'research_artifacts'/f"r64_{member['name']}_{stage}"/'oof.npz'
        z=np.load(f)
        assert all(np.array_equal(z[k],base[k]) for k in ('id','target','fold'))
        predictions.append(z['pred'].astype('float64'))
    mean=np.mean(predictions,axis=0);candidate=pool(tree['tree'],mean)
    out=ROOT/'research_artifacts'/f'r64_viewbag_{stage}';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=mean,candidate=candidate,reference_v11=base['candidate'])
    proof=evidence(base['target'],base['candidate'],candidate,base['fold'])
    result=dict(status='complete',stage=stage,completed_utc=now(),plan_sha256=sha(ROOT/'research_r64_plan.json'),vs_formal_v11=proof,eligible=proof['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'),adaptive_history_acknowledged=True,no_subset_or_weight_search=True)
    dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main(sys.argv[1])

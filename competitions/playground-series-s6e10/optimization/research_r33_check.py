"""One fixed legacy-tree ablation: conditional LGB84 plus current conditional NN74."""
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r15_loader import pool
import json,numpy as np

def main():
    plan=json.loads((ROOT/'research_r33_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    results={}
    for stage in ['fivefold','threefold']:
        tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz')
        assert all(np.array_equal(tree[k],nn[k]) for k in ['id','target','fold'])
        candidate=pool(tree['pred'],nn['pred']);a=evidence(nn['target'],nn['candidate'],candidate,nn['fold']);r=dict(stage=stage,vs_formal_v11=a,eligible=a['eligible_for_new_partition_replication'])
        results[stage]=r;print(stage,a['pooled'],a['corrected_resampled_t'],flush=True)
        if not r['eligible']:break
    eligible=len(results)==2 and all(r['eligible'] for r in results.values())
    result=dict(status='complete',checked_utc=now(),eligible=eligible,validation=results,no_new_training=True,new_formal_submissions=0,csv_generated=False,plan_sha256=sha(ROOT/'research_r33_plan.json'),limitation=plan['historical_label_reuse'])
    dump(ROOT/'research_r33_result.json',result)
if __name__=='__main__':main()

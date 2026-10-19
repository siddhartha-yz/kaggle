"""All leaf contexts of all four routers and every fold before aggregate quality."""
from research_r4_common import ROOT,now,dump,sha,evidence
from scipy.special import expit,logit
import json,numpy as np,sys

def main(stage):
    plan=json.loads((ROOT/'research_r66_plan.json').read_text());base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');fold=base['fold'];pred=np.full(len(fold),np.nan);out=ROOT/f'research_artifacts/r66_routed_tabpfn_{stage}'
    for k in sorted(np.unique(fold)):
        vi=np.flatnonzero(fold==k);routed=[];src=out/f'fold{k}';manifest=json.loads((src/'routing_manifest.json').read_text());assert manifest['status']=='complete' and manifest['plan_sha256']==sha(ROOT/'research_r66_plan.json')
        for row in manifest['routers']:
            p=np.full(len(fold),np.nan);cover=np.zeros(len(fold),'int8')
            for leaf in row['leaves']:
                run=src/f"router{row['router']}/leaf{leaf['leaf']}";r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r66_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r66_worker.py'] and r['validation_sha256']==sha(run/'validation.npz') and r['context_sha256']==sha(run/'context.npz') and r['selection_sha256']==sha(run/'selection.npz')
                selection=np.load(run/'selection.npz');qi=selection['query_idx'];z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],qi) and np.array_equal(z['id'],base['id'][qi]) and np.array_equal(z['target'],base['target'][qi]) and np.isfinite(z['pred']).all();p[qi]=z['pred'];cover[qi]+=1
            assert (cover[vi]==1).all() and not cover[fold!=k].any() and np.isfinite(p[vi]).all();routed.append(p[vi])
        assert len(routed)==4;pred[vi]=np.mean(routed,axis=0)
    assert np.isfinite(pred).all();candidate=expit(.875*logit(np.clip(base['candidate'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=fold,pred=pred,candidate=candidate,reference_v11=base['candidate']);comparison=evidence(base['target'],base['candidate'],candidate,fold);r=dict(status='complete',checked_utc=now(),stage=stage,eligible=comparison['eligible_for_new_partition_replication'],vs_formal_v11=comparison,plan_sha256=sha(ROOT/'research_r66_plan.json'),oof_sha256=sha(out/'oof.npz'),all_contexts_all_folds_before_quality=True);dump(out/'result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main(sys.argv[1])

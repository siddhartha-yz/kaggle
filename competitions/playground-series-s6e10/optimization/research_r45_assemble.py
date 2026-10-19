"""CPU assembly of all frozen fold checkpoints; no training or GPU context."""
from research_r4_common import ROOT,now,dump,sha
from research_r45_loader import load_split,STAGE,pool
import numpy as np,json

def main():
    plan=json.loads((ROOT/'research_r45_plan.json').read_text());df,raw,y,splits,reference,tree,fold,aux=load_split('oof');out=ROOT/f'research_artifacts/r45_newsource_lean_{STAGE}';pred=np.full(len(df),np.nan)
    for k,(fi,vi) in enumerate(splits):
        run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['runner_sha256']==plan['source_sha256']['research_r45_worker.py'] and rec['plan_sha256']==sha(ROOT/'research_r45_plan.json')
        assert rec['model_sha256']==sha(run/'model.pkl') and rec['transformer_sha256']==sha(run/'transformer.joblib')
        z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);pred[vi]=z['pred']
    assert np.isfinite(pred).all();np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=pred,candidate=pool(tree,pred),reference_v11=reference)
    dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_quality_scoring=True,fresh_GPU_process_per_model=True))
if __name__=='__main__':main()

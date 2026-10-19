"""Assemble calibrated predictions only after all fixed NN folds complete."""
from research_r4_common import ROOT,now,dump,sha
from research_r68_loader import load_split,STAGE
from scipy.special import expit,logit
import numpy as np,json

def main():
    plan=json.loads((ROOT/'research_r68_plan.json').read_text());df,raw,y,splits,reference,tree,fold,aux=load_split('oof');out=ROOT/f'research_artifacts/r68_pure_ranknet_{STAGE}';pred=np.full(len(y),np.nan)
    for k,(fi,vi) in enumerate(splits):
        run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r68_plan.json') and r['runner_sha256']==plan['source_sha256']['research_r68_worker.py']
        for name,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('validation.npz','prediction_sha256'),('calibration.json','calibration_sha256'),('calibration_training.npz','calibration_training_sha256')]:assert sha(run/name)==r[key]
        z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);pred[vi]=z['pred']
    assert np.isfinite(pred).all();candidate=expit(.875*logit(np.clip(reference,1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)));np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=pred,candidate=candidate,reference_v11=reference);dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),no_interim_outer_quality=True))
if __name__=='__main__':main()

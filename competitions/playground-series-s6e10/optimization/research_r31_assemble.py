"""CPU aggregation of all CB84 folds plus diagnostic cached raw57 GPU control."""
from research_r4_common import ROOT,now,dump,sha
from research_r15_loader import pool
import numpy as np,json,sys

def main(stage):
    plan=json.loads((ROOT/'research_r31_plan.json').read_text());base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');old=np.load(ROOT/('research_artifacts/cb_deep_oof/raw_oof.npz' if stage=='fivefold' else 'research_artifacts/r4_baseline_replication3/oof.npz'));assert all(np.array_equal(base[k],nn[k]) for k in ['id','target','fold']) and np.array_equal(old['id'],base['id']);oldcb=old['pred'] if stage=='fivefold' else old['components'][:,3]
    out=ROOT/f'research_artifacts/r31_cb84_{stage}';pred=np.full(len(base['id']),np.nan)
    for k in sorted(np.unique(base['fold'])):
        run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['runner_sha256']==plan['source_sha256']['research_r31_worker.py'] and rec['model_sha256']==sha(run/'model.cbm');v=np.load(run/'validation.npz');vi=np.flatnonzero(base['fold']==k);assert np.array_equal(v['idx'],vi) and np.array_equal(v['id'],base['id'][vi]) and np.array_equal(v['target'],base['target'][vi]);pred[vi]=v['pred']
    assert np.isfinite(pred).all();tree=base['tree']+.25*(pred-oldcb);extra={}
    if stage=='fivefold':
        control=np.load(ROOT/'research_artifacts/r18_gpu_cb_numeric57_fivefold/oof.npz');assert all(np.array_equal(control[k],base[k]) for k in ['id','target','fold']);extra['GPU57_control']=pool(base['tree']+.25*(control['pred']-oldcb),nn['pred'])
    np.savez_compressed(out/'oof.npz',id=base['id'],target=base['target'],fold=base['fold'],pred=pred,tree=tree,candidate=pool(tree,nn['pred']),reference_v11=nn['candidate'],**extra);dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),all_folds_before_scoring=True))
if __name__=='__main__':main(sys.argv[1])

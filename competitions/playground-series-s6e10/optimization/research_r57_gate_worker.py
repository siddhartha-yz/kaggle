"""Fit cohort coefficients only on inner-heldout experts; infer sealed outer experts."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,threading,joblib,numpy as np,pandas as pd
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r57_gate import CohortGate
def main(stage,k):
    plan=json.loads((ROOT/'research_r57_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/f'research_r57_gate_{stage}_{k}_resource_failure.json'),daemon=True);watcher.start()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');run=ROOT/f'research_artifacts/r57_nested_gate_{stage}/fold{k}';idx=np.load(run/'inner_indices.npz');it,im,vi=idx['inner_train'],idx['inner_meta'],idx['outer_valid'];assert not np.intersect1d(np.r_[it,im],vi).size and not np.intersect1d(it,im).size
        treepath=run/'trees';nrpath=ROOT/f'research_artifacts/r57_inner_lean_{stage}/fold{k}';tr=json.loads((treepath/'result.json').read_text());nr=json.loads((nrpath/'result.json').read_text());assert tr['status']=='complete' and nr['status']=='complete' and tr['plan_sha256']==nr['plan_sha256']==sha(ROOT/'research_r57_plan.json') and tr['prediction_sha256']==sha(treepath/'meta_predictions.npz') and nr['model_sha256']==sha(nrpath/'model.pkl')
        trees=np.load(treepath/'meta_predictions.npz');nn=np.load(nrpath/'validation.npz');assert all(np.array_equal(z['idx'],im) and np.array_equal(z['id'],df.id.iloc[im]) and np.array_equal(z['target'],y[im]) for z in [trees,nn]);assert nr['training_index_sha256']==__import__('hashlib').sha256(it.tobytes()).hexdigest()
        inner_tree=trees['components']@np.array([.46875,.09375,.1875,.25]);gate=CohortGate(ridge=100.).fit(raw.iloc[im],inner_tree,nn['pred'],y[im]);joblib.dump(gate,run/'gate.joblib');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');outer=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');assert all(np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) for z in [base,outer]) and np.array_equal(base['fold'],outer['fold']) and np.array_equal(np.flatnonzero(base['fold']==k),vi)
        pred=gate.predict(raw.iloc[vi],outer['tree'][vi],base['pred'][vi]);again=joblib.load(run/'gate.joblib').predict(raw.iloc[vi],outer['tree'][vi],base['pred'][vi]);assert np.array_equal(pred,again) and np.isfinite(pred).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,reference_v11=base['candidate'][vi])
        result=dict(status='complete',completed_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r57_plan.json'),runner_sha256=sha(__file__),inner_indices_sha256=sha(run/'inner_indices.npz'),inner_tree_record_sha256=sha(treepath/'result.json'),inner_nn_record_sha256=sha(nrpath/'result.json'),inner_nn_prediction_sha256=sha(nrpath/'validation.npz'),gate_sha256=sha(run/'gate.joblib'),validation_sha256=sha(run/'validation.npz'),cohort_vocabulary=gate.vocab.tolist(),delta_from_equal_weight=gate.theta.tolist(),regularization=100.,optimization=gate.optimization,inner_meta_rows=len(im),outer_validation_rows=len(vi),outer_validation_excluded_from_all_fits=True,no_outer_quality_scoring=True,peak_RSS_bytes=peak[0]);dump(run/'result.json',result);print('OUTER_FOLD_READY_NO_QUALITY',stage,k,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

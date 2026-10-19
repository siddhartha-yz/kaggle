"""Confidence gate fit only on inner-heldout expert predictions, sealed outer inference."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,threading,hashlib,joblib,numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r65_gate import ConfidenceGate

def main(stage,k):
    plan=json.loads((ROOT/'research_r65_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r65_confidence_gate_fivefold/result.json').read_text())['eligible']
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/f'research_r65_gate_{stage}_{k}_resource_failure.json'),daemon=True);watcher.start()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');outer=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
        assert all(np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) for z in [base,outer]) and np.array_equal(base['fold'],outer['fold']);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020)
        route='r57' if stage=='fivefold' else 'r65';directory='r57_nested_gate' if stage=='fivefold' else 'r65_confidence_gate';src=ROOT/f'research_artifacts/{directory}_{stage}/fold{k}';idx=np.load(src/'inner_indices.npz');assert all(np.array_equal(idx[n],v) for n,v in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)]) and not np.intersect1d(np.r_[it,im],vi).size
        treepath=src/'trees';nrpath=ROOT/f'research_artifacts/{route}_inner_lean_{stage}/fold{k}';tr=json.loads((treepath/'result.json').read_text());nr=json.loads((nrpath/'result.json').read_text());assert tr['status']=='complete' and nr['status']=='complete' and tr['plan_sha256']==nr['plan_sha256']==sha(ROOT/f'research_{route}_plan.json') and tr['prediction_sha256']==sha(treepath/'meta_predictions.npz') and nr['model_sha256']==sha(nrpath/'model.pkl') and nr['prediction_sha256']==sha(nrpath/'validation.npz') and nr['training_index_sha256']==hashlib.sha256(it.tobytes()).hexdigest()
        trees=np.load(treepath/'meta_predictions.npz');nn=np.load(nrpath/'validation.npz');assert all(np.array_equal(z['idx'],im) and np.array_equal(z['id'],df.id.iloc[im]) and np.array_equal(z['target'],y[im]) for z in [trees,nn])
        inner_tree=trees['components']@np.array(plan['tree_weights']);gate=ConfidenceGate(ridge=100.).fit(inner_tree,nn['pred'],y[im]);run=ROOT/f'research_artifacts/r65_confidence_gate_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True);joblib.dump(gate,run/'gate.joblib');pred=gate.predict(outer['tree'][vi],base['pred'][vi]);again=joblib.load(run/'gate.joblib').predict(outer['tree'][vi],base['pred'][vi]);assert np.array_equal(pred,again) and np.isfinite(pred).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,reference_v11=base['candidate'][vi])
        record=dict(status='complete',completed_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r65_plan.json'),runner_sha256=sha(__file__),inner_source_route=route,inner_indices_sha256=sha(src/'inner_indices.npz'),inner_tree_record_sha256=sha(treepath/'result.json'),inner_nn_record_sha256=sha(nrpath/'result.json'),inner_nn_prediction_sha256=sha(nrpath/'validation.npz'),gate_sha256=sha(run/'gate.joblib'),validation_sha256=sha(run/'validation.npz'),confidence_coefficients=gate.beta.tolist(),regularization=100.,optimization=gate.optimization,inner_meta_rows=len(im),outer_validation_rows=len(vi),outer_validation_excluded_from_all_fits=True,no_outer_quality_scoring=True,peak_RSS_bytes=peak[0]);dump(run/'result.json',record);print('OUTER_FOLD_READY_NO_QUALITY',stage,k,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

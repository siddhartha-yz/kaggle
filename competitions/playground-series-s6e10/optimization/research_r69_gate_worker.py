"""Fit one scalar on proper inner-heldout three-expert scores, then seal outer inference."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,threading,hashlib,joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.model_selection import train_test_split
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r69_gate import PairGate

def main(stage,k):
    plan=json.loads((ROOT/'research_r69_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r69_rank_gate_fivefold/result.json').read_text())['eligible']
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/f'research_r69_gate_{stage}_{k}_resource_failure.json'),daemon=True);watcher.start()
    try:
        df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');outer=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');rank=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz')
        assert all(np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],base['fold']) for z in [base,outer,rank])
        fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);run=ROOT/f'research_artifacts/r69_rank_gate_{stage}/fold{k}';idx=np.load(run/'inner_indices.npz');assert all(np.array_equal(idx[n],v) for n,v in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)]) and not np.intersect1d(np.r_[it,im],vi).size
        route='r57' if stage=='fivefold' else 'r69';directory='r57_nested_gate' if stage=='fivefold' else 'r69_rank_gate';src=ROOT/f'research_artifacts/{directory}_{stage}/fold{k}';treepath=src/'trees';nrpath=ROOT/f'research_artifacts/{route}_inner_lean_{stage}/fold{k}';rrpath=ROOT/f'research_artifacts/r69_inner_ranknet_{stage}/fold{k}'
        tr=json.loads((treepath/'result.json').read_text());nr=json.loads((nrpath/'result.json').read_text());rr=json.loads((rrpath/'result.json').read_text());assert tr['status']==nr['status']==rr['status']=='complete' and tr['plan_sha256']==nr['plan_sha256']==sha(ROOT/f'research_{route}_plan.json') and rr['plan_sha256']==sha(ROOT/'research_r69_plan.json')
        assert tr['prediction_sha256']==sha(treepath/'meta_predictions.npz') and nr['model_sha256']==sha(nrpath/'model.pkl') and nr['prediction_sha256']==sha(nrpath/'validation.npz') and rr['model_sha256']==sha(rrpath/'model.pkl') and rr['prediction_sha256']==sha(rrpath/'validation.npz') and rr['calibration_sha256']==sha(rrpath/'calibration.json')
        assert nr['training_index_sha256']==rr['training_index_sha256']==hashlib.sha256(it.tobytes()).hexdigest()
        trees=np.load(treepath/'meta_predictions.npz');nn=np.load(nrpath/'validation.npz');rn=np.load(rrpath/'validation.npz');assert all(np.array_equal(z['idx'],im) and np.array_equal(z['id'],df.id.iloc[im]) and np.array_equal(z['target'],y[im]) for z in [trees,nn,rn])
        inner_tree=trees['components']@np.array(plan['tree_weights']);gate=PairGate(**plan['gate_parameters']).fit(inner_tree,nn['pred'],rn['pred'],y[im]);joblib.dump(gate,run/'gate.joblib')
        zero=expit(.875*.5*(logit(np.clip(outer['tree'][vi],1e-6,1-1e-6))+logit(np.clip(base['pred'][vi],1e-6,1-1e-6)))+.125*logit(np.clip(rank['pred'][vi],1e-6,1-1e-6)));assert np.max(np.abs(zero-rank['candidate'][vi]))<1e-12
        pred=gate.predict(outer['tree'][vi],base['pred'][vi],rank['pred'][vi]);again=joblib.load(run/'gate.joblib').predict(outer['tree'][vi],base['pred'][vi],rank['pred'][vi]);assert np.array_equal(pred,again) and np.isfinite(pred).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,reference_v12=rank['candidate'][vi])
        record=dict(status='complete',completed_utc=now(),stage=stage,fold=k,plan_sha256=sha(ROOT/'research_r69_plan.json'),runner_sha256=sha(__file__),inner_source_route=route,inner_indices_sha256=sha(run/'inner_indices.npz'),inner_tree_record_sha256=sha(treepath/'result.json'),inner_nn_record_sha256=sha(nrpath/'result.json'),inner_ranknet_record_sha256=sha(rrpath/'result.json'),gate_sha256=sha(run/'gate.joblib'),validation_sha256=sha(run/'validation.npz'),scalar_delta=gate.theta,optimization=gate.optimization,inner_meta_rows=len(im),outer_validation_rows=len(vi),outer_validation_excluded_from_all_fits=True,no_outer_quality_scoring=True,peak_RSS_bytes=peak[0]);dump(run/'result.json',record);print('OUTER_FOLD_READY_NO_QUALITY',stage,k,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

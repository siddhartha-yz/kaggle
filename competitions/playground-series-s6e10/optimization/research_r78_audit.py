"""Separate postqualification CPU audit; no fit, quality search, or recipe changes."""
from research_r78_model import ROOT,sha,np,torch,auxiliary,references,candidate,config
from research_r4_common import now,dump,evidence
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures
from research_r78_features import reconstruct_states,H,EPS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from threadpoolctl import threadpool_limits
import os,json,threading,time,gc,psutil,cloudpickle,joblib,pandas as pd
def monitor(stop,peak):
 while not stop.wait(.25):
  rss=psutil.Process().memory_info().rss;peak[0]=max(peak[0],rss);available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
  if rss>8*2**30 or available<4*2**30:dump(ROOT/'research_r78_audit_resource_failure.json',dict(RSS=rss,available=available));os._exit(86)
def main():
 torch.set_num_threads(6);torch.set_num_interop_threads(1);p=json.loads((ROOT/'research_r78_plan.json').read_text());ap=json.loads((ROOT/'research_r78_audit_plan.json').read_text());ph=sha(ROOT/'research_r78_plan.json');assert ap['main_plan_sha256']==ph and ap['audit_source_sha256']==sha(__file__)
 for n,h in {**p['source_sha256'],**p['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 assert json.loads((ROOT/'research_r78_driver_state.json').read_text())['status']=='qualified_requires_independent_audit';peak=[0];stop=threading.Event();threading.Thread(target=monitor,args=(stop,peak),daemon=True).start()
 try:
  prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65;src=ROOT/'research_artifacts/r20_aux_officialcb';m=json.loads((src/'result.json').read_text());assert len(m['models'])==65 and not m['satisfaction_column_loaded'] and not m['satisfaction_used_as_target_or_input']
  for row in m['models']:assert sha(src/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
  frame=pd.read_csv(ROOT/'data/train.csv');assert len(frame)==699635 and frame.satisfaction.dtype==bool;ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);a=auxiliary(ids);rows=[];proofs={}
  for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
   out=ROOT/f'research_artifacts/r78_logratio_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base,tree=references(stage);pred=np.full(len(y),np.nan);fold=np.full(len(y),-1,'int8')
   for k,(fi,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):
    fold[vi]=k;run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==ph and rec['config']==config() and not rec['validation_used_for_fit_or_epoch_selection'];assert all(sha(run/n)==h for n,h in rec['files_sha256'].items());tr=joblib.load(run/'transformer.joblib');assert tr.columns==list(raw) and tr.aux_columns==list(a) and tr.encoder.cv==5 and tr.encoder.random_state==42
    cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True),cols) for j,cols in enumerate(COMBOS)})
    with threadpool_limits(6):enc=TargetEncoder(**tr.encoder.get_params());te=enc.fit_transform(cross,y[fi])
    savedte=np.load(run/'training_te.npz');assert np.array_equal(savedte['idx'],fi) and np.array_equal(savedte['values'],te.astype('float32'));assert enc.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(x,b) for x,b in zip(enc.categories_,tr.encoder.categories_)) and all(np.array_equal(x,b) for x,b in zip(enc.encodings_,tr.encoder.encodings_));del cross,enc,te,savedte
    for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
    ix=np.load(run/'indices.npz');saved=np.load(run/'validation.npz')
    for n,v in [('fi',fi),('vi',vi),('train_id',ids[fi]),('valid_id',ids[vi])]:assert np.array_equal(ix[n],v)
    assert all(np.array_equal(saved[n],v) for n,v in [('idx',vi),('id',ids[vi]),('target',y[vi])]);query=vi[:2048];V=tr.transform(raw.iloc[query].reset_index(drop=True),a.iloc[query].reset_index(drop=True));old=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');C=old.transform(raw.iloc[query].reset_index(drop=True),a.iloc[query,:27].reset_index(drop=True));pd.testing.assert_frame_equal(V.iloc[:,:74],C,check_exact=True);assert V.shape==(len(query),139) and V.select_dtypes(include='number').shape[1]==117
    # Independently reconstruct fixed coordinate axes from stored state probabilities.
    cached=np.load(ROOT/'research_artifacts/r25_posterior_officialcb/train_features.npz');prob=np.maximum(cached['values'][query].reshape(-1,13,6).astype('float64'),EPS);prob/=prob.sum(-1,keepdims=True);expected=(np.log(prob)@H.T).reshape(-1,65).astype('float32');assert np.array_equal(expected,V.iloc[:,-65:].to_numpy());inverse_error=float(np.max(np.abs(reconstruct_states(expected)-prob)));assert inverse_error<1e-6;del cached,prob,expected
    with (run/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
    module=model.cv_alg_interface_.model;assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and module.progress.max_epochs==3 and not module.creator.is_cv and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu'
    with threadpool_limits(6):replay=model.predict_proba(V)[:,1]
    error=float(np.max(np.abs(replay-saved['pred'][:len(query)])));assert error<2e-6;pred[vi]=saved['pred'];rows.append(dict(stage=stage,fold=k,model_sha256=rec['files_sha256']['model.pkl'],native_members=32,strict_OOF_train_TE_exact=True,original74_query_fields_exact=True,own_outer_train_counts_exact=True,CPU_replay_error=error,logratio_inverse_error=inverse_error));print('AUDITED',stage,k,error,flush=True);del tr,old,C,V,model,module,saved;gc.collect()
   assert np.array_equal(z['id'],ids) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold) and np.array_equal(base['fold'],fold) and np.array_equal(pred,z['pred']);pool=candidate(tree,pred,base['pred']);assert np.array_equal(pool,z['candidate']) and np.array_equal(z['reference_v12'],base['candidate']);proof=evidence(y,base['candidate'],pool,fold);assert proof['eligible_for_new_partition_replication'];proofs[stage]=proof
  dump(ROOT/'research_r78_pre_inference_audit.json',dict(status='passed',audited_utc=now(),plan_sha256=ph,audit_plan_sha256=sha(ROOT/'research_r78_audit_plan.json'),verified_model_records=8,verified_native_members=256,model_records=rows,validation=proofs,rechecked_auxiliary_models=65,positive_class='True=1',peak_RSS_bytes=peak[0],statistics_limitation=p['statistics_limitation']))
 finally:stop.set()
if __name__=='__main__':main()

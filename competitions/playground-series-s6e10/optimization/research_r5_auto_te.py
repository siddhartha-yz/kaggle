"""Fixed v7 component contrast; complete fivefold then cached independent-fit threefold."""
from research_r4_common import ROOT,now,dump,sha,guard,evidence
import os,json,time,gc,joblib,resource
import numpy as np,pandas as pd,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_next_auto_te_features import ConditionalAutoTE
from research_external_ablation import external
from research_oof import params
W=45/128

def data(stage):
 df=pd.read_csv(ROOT/'data/train.csv');assert sha(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36';y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction'])
 if stage=='fivefold':
  z=np.load(ROOT/'research_artifacts/r4_lean_gpu_init_oof/oof.npz');o=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz');assert np.array_equal(z['id'],o['id']) and np.array_equal(z['fold'],o['fold']);old=o['pred']
 else:
  z=np.load(ROOT/'research_artifacts/r4_lean_gpu_init_replication3/oof.npz');o=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz');assert np.array_equal(z['id'],o['id']) and np.array_equal(z['fold'],o['fold']);old=o['components'][:,0]
 assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);return df,raw,y,z['fold'],z['candidate'],old

def fit(stage):
 guard();out=ROOT/'research_artifacts'/('r5_auto_te_'+stage);out.mkdir(exist_ok=True);df,raw,y,fold,ref,old=data(stage);ext,_=external('both',df.id);K=len(np.unique(fold));predall=np.full(len(y),np.nan);plan=ROOT/'research_r5_plan.json'
 for k in range(K):
  fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';run.mkdir(exist_ok=True);result=run/'result.json';tick=time.monotonic()
  if result.exists() and json.loads(result.read_text()).get('status')=='complete':
   rec=json.loads(result.read_text());assert rec['model_sha256']==sha(run/'model.txt') and rec['runner_sha256']==sha(__file__);z=np.load(run/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);predall[vi]=z['pred'];continue
  rec={'status':'training','started_utc':now(),'stage':stage,'fold':k,'params':params('stats639'),'plan_sha256':sha(plan),'runner_sha256':sha(__file__),'features_sha256':sha(ROOT/'research_next_auto_te_features.py'),'train_rows':len(fi),'valid_rows':len(vi),'train_index_sha256':__import__('hashlib').sha256(fi.tobytes()).hexdigest(),'main_target_encoding':'inner5foldonlyauto21raw+13segmentresponses','validation_features_and_targets_excluded_from_all_fits':True};dump(result,rec);dump(ROOT/'research_r5_state.json',dict(rec,active_training=True,pid=os.getpid()));print('START',stage,k,flush=True)
  try:
   with threadpool_limits(6):
    tr=ConditionalAutoTE();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));assert X.shape[1]==91;joblib.dump(tr,run/'transformer.joblib');m=lgb.LGBMClassifier(**params('stats639'));m.fit(X,y[fi]);p=m.predict_proba(V)[:,1];m.booster_.save_model(str(run/'model.txt'));loaded=lgb.Booster(model_file=str(run/'model.txt'));again=loaded.predict(V,num_threads=6);err=float(np.max(np.abs(p-again)));assert err<1e-12 and np.isfinite(p).all()
   predall[vi]=p;candidate=ref[vi]+W*(p-old[vi]);assert (candidate>=0).all() and (candidate<=1).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p,candidate=candidate,reference_v7=ref[vi]);rec.update(status='complete',ended_utc=now(),feature_width=91,seconds=time.monotonic()-tick,model_sha256=sha(run/'model.txt'),prediction_sha256=sha(run/'validation.npz'),transformer_sha256=sha(run/'transformer.joblib'),reload_error=err,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(result,rec);print('DONE',stage,k,rec['seconds'],flush=True);del tr,X,V,m,loaded;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(result,rec);raise
 assert np.isfinite(predall).all();candidate=ref+W*(predall-old);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=candidate,reference_v7=ref,old_lgb=old);res=evidence(y,ref,candidate,fold);res.update(status='complete',ended_utc=now(),route='conditional_auto_te',stage=stage,oof_sha256=sha(out/'oof.npz'),plan_sha256=sha(plan));dump(out/'result.json',res);dump(ROOT/'research_r5_state.json',{'updated_utc':now(),'status':'stage_complete','stage':stage,'active_training':False,'qualified':res['eligible_for_new_partition_replication']});print(json.dumps(res,indent=2),flush=True);return res

def main():
 r=fit('fivefold')
 if r['eligible_for_new_partition_replication']:r=fit('threefold')
 dump(ROOT/'research_r5_auto_te_result.json',{'updated_utc':now(),'status':'complete','eligible_for_candidate':r['stage']=='threefold' and r['eligible_for_new_partition_replication'],'last_stage':r['stage'],'actual_top100_verified':False,'formal_new_used':1,'remaining':1})
if __name__=='__main__':main()

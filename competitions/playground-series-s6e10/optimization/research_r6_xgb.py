"""One higher-order conditional response learner; nested preprocessing and tree-count selection."""
from research_r4_common import ROOT,now,dump,sha,guard,evidence
import os,time,gc,json,argparse,joblib,resource
import numpy as np,xgboost as xgb
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from research_next_auto_te_features import ConditionalAutoTE
from research_external_ablation import external
from research_r5_auto_te import data,W
PARAMS=dict(n_estimators=6000,learning_rate=.03,max_depth=8,min_child_weight=5,reg_lambda=5.,reg_alpha=.05,subsample=.9,colsample_bytree=.5,max_bin=256,objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cpu',enable_categorical=True,max_cat_to_onehot=8,n_jobs=6,random_state=20261024)
class ResourceCallback(xgb.callback.TrainingCallback):
 def __init__(self,path,phase):self.path=path;self.phase=phase;self.start=time.monotonic()
 def after_iteration(self,model,epoch,evals_log):
  if epoch%100==0:
   guard();elapsed=time.monotonic()-self.start;assert elapsed<=1800,'Singlefitexceededfrozen1800sresourcebound';assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'RSSabove8GiB';dump(self.path,{'updated_utc':now(),'phase':self.phase,'round':epoch+1,'seconds':elapsed,'pid':os.getpid(),'active_training':True,'CPU_threads':6});print('ITER',self.phase,epoch+1,elapsed,flush=True)
  return False

def fit(stage,probe=False):
 df,raw,y,fold,ref,old=data(stage);ext,_=external('both',df.id);out=ROOT/'research_artifacts'/('r6_xgb_probe' if probe else 'r6_xgb_'+stage);out.mkdir(exist_ok=True);predall=np.full(len(y),np.nan)
 for k in range(len(np.unique(fold))):
  guard();fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261024);assert not np.intersect1d(it,im).size and not np.intersect1d(fi,vi).size;run=out/f'fold{k}';run.mkdir(exist_ok=True);result=run/'result.json'
  if not probe and result.exists() and json.loads(result.read_text()).get('status')=='complete':
   rec=json.loads(result.read_text());assert rec['model_sha256']==sha(run/'model.ubj') and rec['runner_sha256']==sha(__file__);z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['idx'],vi);predall[vi]=z['pred'];continue
  np.savez_compressed(run/'indices.npz',outer_train=fi,outer_valid=vi,inner_train=it,inner_meta=im);rec={'status':'inner_preprocessing','started_utc':now(),'stage':stage,'fold':k,'params':PARAMS,'runner_sha256':sha(__file__),'plan_sha256':sha(ROOT/'research_r6_plan.json'),'indices_sha256':sha(run/'indices.npz'),'nested_TE_and_distance_maps_refitted_on_inner_train':True,'outer_valid_never_used_for_selection':True};dump(result,rec);tick=time.monotonic();print('START',stage,k,'probe' if probe else 'nested',flush=True)
  with threadpool_limits(6):
   tr=ConditionalAutoTE();X=tr.fit_transform(raw.iloc[it].reset_index(drop=True),y[it],ext.iloc[it].reset_index(drop=True));cfg=dict(PARAMS,n_estimators=50) if probe else dict(PARAMS,early_stopping_rounds=200);cfg['callbacks']=[ResourceCallback(ROOT/'research_r6_state.json','resource_probe' if probe else f'{stage}_fold{k}_inner')];m=xgb.XGBClassifier(**cfg);rec['status']='resource_probe' if probe else 'inner_selection';dump(result,rec)
   if probe:
    start=time.monotonic();m.fit(X,y[it],verbose=False);rec.update(status='complete_no_quality_score',fit_seconds=time.monotonic()-start,feature_width=X.shape[1],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,score_evaluated=False,model_rounds=m.get_booster().num_boosted_rounds());dump(result,rec);dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);return rec
   V=tr.transform(raw.iloc[im].reset_index(drop=True),ext.iloc[im].reset_index(drop=True));m.fit(X,y[it],eval_set=[(V,y[im])],verbose=100);best=int(m.best_iteration)+1;assert 1<=best<=6000;rec.update(selected_rounds=best,inner_best_auc=float(m.best_score),inner_fitted_rounds=m.get_booster().num_boosted_rounds(),selection_at_cap=best>5800,status='outer_refit');dump(result,rec);del m,X,V,tr;gc.collect()
   tr=ConditionalAutoTE();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib');cfg=dict(PARAMS,n_estimators=best,callbacks=[ResourceCallback(ROOT/'research_r6_state.json',f'{stage}_fold{k}_outer')]);m=xgb.XGBClassifier(**cfg);m.fit(X,y[fi],verbose=False);assert m.get_booster().num_boosted_rounds()==best and np.array_equal(m.classes_,[0,1]);p=m.predict_proba(V)[:,1];m.save_model(str(run/'model.ubj'));loaded=xgb.XGBClassifier(n_jobs=6,device='cpu');loaded.load_model(str(run/'model.ubj'));loaded.get_booster().set_param({'nthread':6,'device':'cpu'});again=loaded.predict_proba(V)[:,1];err=float(np.max(np.abs(p-again)));assert err<1e-7 and np.isfinite(p).all()
  predall[vi]=p;candidate=ref[vi]+W*(p-old[vi]);assert (candidate>=0).all() and (candidate<=1).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=p,candidate=candidate,reference_v7=ref[vi]);rec.update(status='complete',ended_utc=now(),seconds=time.monotonic()-tick,feature_width=X.shape[1],model_sha256=sha(run/'model.ubj'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),reload_error=err,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(result,rec);print('DONE',stage,k,rec['seconds'],flush=True);del m,loaded,X,V,tr;gc.collect()
 candidate=ref+W*(predall-old);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=candidate,reference_v7=ref,old_lgb=old);res=evidence(y,ref,candidate,fold);res.update(status='complete',stage=stage,ended_utc=now(),route='nested_deep_xgb',oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',res);dump(ROOT/'research_r6_state.json',{'updated_utc':now(),'status':'stage_complete','stage':stage,'active_training':False,'qualified':res['eligible_for_new_partition_replication']});print(json.dumps(res,indent=2),flush=True);return res

def main():
 a=argparse.ArgumentParser();a.add_argument('--probe',action='store_true');a=a.parse_args();r=fit('fivefold',a.probe)
 if a.probe:return
 if r['eligible_for_new_partition_replication']:r=fit('threefold')
 dump(ROOT/'research_r6_result.json',{'status':'complete','updated_utc':now(),'eligible_for_candidate':r['stage']=='threefold' and r['eligible_for_new_partition_replication'],'last_stage':r['stage'],'formal_used':1,'formal_remaining':1,'actualTop100verified':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_r6_failure.json',{'updated_utc':now(),'status':'failed','error':repr(e)});raise

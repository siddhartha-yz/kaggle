"""Privileged original service supervision, marginalized for every official competition row."""
from research_w3_common import *
import gc,time,joblib,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_w3_service_source import safe_service_source
from research_original import OriginalLookup
from research_features import DistanceFeatures
def with_service(base,value,classes):
 out=base.copy();v=np.full(len(base),int(value),dtype='int8') if np.isscalar(value) else np.asarray(value,dtype='int8');out['latent_service_value']=v.astype('float32');out['latent_service_category']=pd.Categorical(v.astype(str),categories=[str(int(c)) for c in classes]);return out
def features(raw,tr,svc,sat,classes,lookup):
 X=tr.transform_te(raw);p=svc.predict(X,num_threads=6);assert p.shape==(len(raw),len(classes));probs=np.zeros((len(raw),6))
 for j,c in enumerate(classes):probs[:,int(c)]=p[:,j]
 assert np.allclose(probs.sum(1),1,rtol=0,atol=1e-7);marginal=np.zeros(len(raw))
 for j,c in enumerate(classes):
  for lo in range(0,len(raw),65536):
   hi=min(lo+65536,len(raw));q=sat.predict(with_service(X.iloc[lo:hi],c,classes),num_threads=6);marginal[lo:hi]+=p[lo:hi,j]*q
 out={f'svc_p{c}':probs[:,c] for c in range(6)};out['svc_expected']=probs@np.arange(6);out['svc_entropy']=-np.sum(probs*np.log(np.clip(probs,1e-12,1)),axis=1);near=lookup.transform(raw);out['svc_distance_mean']=near.orig_lookup0_mean.to_numpy();out['svc_near_mean']=near.orig_lookup1_mean.to_numpy();out['svc_marginal_sat']=marginal;result=pd.DataFrame(out).astype('float32');assert np.isfinite(result.to_numpy()).all() and result.svc_marginal_sat.between(0,1).all();return result
def main():
 start_guard();out=ROOT/'research_artifacts/w3_service_teacher';out.mkdir(exist_ok=False);source,train,test,cols,audit=safe_service_source();raw=source[cols];service=source['Inflight service'].astype('int8').to_numpy();target=source.satisfaction.astype('int8').to_numpy();classes=np.sort(np.unique(service));rec={'status':'source_service_training','started_utc':now(),'source_audit':audit,'plan_sha256':sha(ROOT/'research_w3_service_plan.json'),'runner_sha256':sha(__file__),'source_helper_sha256':sha(ROOT/'research_w3_service_source.py'),'service_classes':classes.tolist(),'competition_targets_used_or_read':False};dump(out/'result.json',rec);params=dict(n_estimators=475,learning_rate=.04,num_leaves=63,min_child_samples=100,reg_lambda=10,n_jobs=6,random_state=20261018,verbosity=-1,deterministic=True,force_col_wise=True)
 try:
  with threadpool_limits(6):
   tr=DistanceFeatures('base');X=tr.fit_transform_te(raw,service);assert not tr.te_maps;svc=lgb.LGBMClassifier(**params,objective='multiclass');svc.fit(X,service);assert np.array_equal(svc.classes_,classes);svc.booster_.save_model(str(out/'service_model.txt'));sbo=lgb.Booster(model_file=str(out/'service_model.txt'));rd=float(np.max(np.abs(svc.predict_proba(X.iloc[:1003])-sbo.predict(X.iloc[:1003],num_threads=6))));assert rd<1e-12;rec.update(status='source_SAT22_training',service_model_reload_max_abs_diff=rd);dump(out/'result.json',rec);print(rec['status'],flush=True);start_guard();XX=with_service(X,service,classes);sat=lgb.LGBMClassifier(**params,objective='binary');sat.fit(XX,target);assert np.array_equal(sat.classes_,[0,1]);sat.booster_.save_model(str(out/'sat22_model.txt'));sabo=lgb.Booster(model_file=str(out/'sat22_model.txt'));rd2=float(np.max(np.abs(sat.predict_proba(XX.iloc[:1003])[:,1]-sabo.predict(XX.iloc[:1003],num_threads=6))));assert rd2<1e-12;lookup=OriginalLookup().fit(raw,service);joblib.dump(tr,out/'transformer.joblib');joblib.dump(lookup,out/'service_lookup.joblib');rec.update(status='predicting_competition_features',SAT22_model_reload_max_abs_diff=rd2);dump(out/'result.json',rec);del svc,sat,X,XX;gc.collect()
   for label,df in [('train',train),('test',test)]:
    result=features(df[cols].reset_index(drop=True),tr,sbo,sabo,classes,lookup);assert len(result)==len(df) and result.shape[1]==11;np.savez_compressed(out/f'{label}_features.npz',id=df.id.to_numpy(),values=result.to_numpy(),columns=np.asarray(result.columns,dtype=str));rec[label+'_features_sha256']=sha(out/f'{label}_features.npz');dump(out/'result.json',rec);print('SERVICE_FEATURES',label,len(result),flush=True)
  rec.update(status='complete',completed_utc=now(),service_model_sha256=sha(out/'service_model.txt'),sat22_model_sha256=sha(out/'sat22_model.txt'),transformer_sha256=sha(out/'transformer.joblib'),lookup_sha256=sha(out/'service_lookup.joblib'));dump(out/'result.json',rec)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(out/'result.json',rec);raise
if __name__=='__main__':main()

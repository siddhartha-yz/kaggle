import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,resource,joblib,numpy as np,pandas as pd,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_r90_common import *
from research_r90_features import LowCardinalRatingCategories
from research_r23_loader import auxiliary
from research_external_ablation import external
from research_oof import params
from research_resume_policy_20261021 import check_training

def main():
 stage=sys.argv[1];k=int(sys.argv[2]);plan=verify();check_training();z=refs(stage);r=z[0]
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction'])
 assert np.array_equal(df.id,r['id']) and np.array_equal(y,r['target'])
 fi=np.flatnonzero(r['fold']!=k);vi=np.flatnonzero(r['fold']==k)
 ext,_=external('both',df.id);aux=auxiliary(df.id)
 run=OUT/stage/f'fold{k}';run.mkdir(parents=True,exist_ok=True);rec=run/'result.json'
 if rec.exists():
  old=json.loads(rec.read_text());assert old['status']=='complete' and old['plan_sha256']==sha(ROOT/'research_r90_plan.json')
  for n in ('model.txt','transformer.joblib','validation.npz'):assert sha(run/n)==old['artifact_sha256'][n]
  print('REUSED',stage,k,flush=True);return
 tick=time.monotonic();state=dict(status='training',stage=stage,fold=k,started_utc=now(),pid=os.getpid(),plan_sha256=sha(ROOT/'research_r90_plan.json'),own_outer_training_rows=len(fi),validation_rows=len(vi),validation_used_for_fit=False,parameters=params('stats639'))
 dump(rec,state);print('START',stage,k,state['pid'],flush=True)
 take=lambda x,ix:x.iloc[ix].reset_index(drop=True)
 with threadpool_limits(6):
  tr=LowCardinalRatingCategories();X=tr.fit_transform(take(raw,fi),y[fi],take(ext,fi),take(aux,fi));V=tr.transform(take(raw,vi),take(ext,vi),take(aux,vi))
  old=joblib.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib')
  pd.testing.assert_frame_equal(V.iloc[:,:84],old.transform(take(raw,vi),take(ext,vi),take(aux,vi)),check_exact=True)
  joblib.dump(tr,run/'transformer.joblib',compress=3)
  model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y[fi]);assert np.array_equal(model.classes_,[0,1]);pred=model.predict_proba(V)[:,1]
  model.booster_.save_model(str(run/'model.txt'));loaded=lgb.Booster(model_file=str(run/'model.txt'));err=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))));assert err<1e-12
 assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
 np.savez_compressed(run/'validation.npz',id=df.id.to_numpy()[vi],target=y[vi],idx=vi,pred=pred)
 state.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,features=93,unchanged_first84_exact=True,model_reload_error=err,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,artifact_sha256={n:sha(run/n) for n in ('model.txt','transformer.joblib','validation.npz')})
 dump(rec,state);print('DONE',stage,k,state['seconds'],flush=True)
if __name__=='__main__':main()

"""Original-only cross-fitted auxiliary features for joint source/synthetic supervision."""
from research_w3_common import *
import joblib,gc,time,lightgbm as lgb
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
from research_original import safe_original,OriginalLookup,normalized_hash
from research_features import DistanceFeatures
def main():
 start_guard();pp=json.loads((ROOT/'research_w3_augmentation_plan.json').read_text());out=ROOT/'research_artifacts/w3_source_aux_oof';out.mkdir(exist_ok=False);source,_,_,cols,audit=safe_original();raw=source[cols];y=source.satisfaction.astype('int8').to_numpy();rowhash=normalized_hash(source,cols);fold=np.full(len(y),-1,dtype='int8');values=np.full((len(y),5),np.nan);columns=['orig_lookup0_mean','orig_lookup0_logcount','orig_lookup1_mean','orig_lookup1_logcount','orig_teacher_lgb'];rec={'status':'running','started_utc':now(),'source_audit':audit,'synthetic_targets_used':False,'plan_sha256':sha(ROOT/'research_w3_augmentation_plan.json'),'runner_sha256':sha(__file__),'folds':[]};dump(out/'result.json',rec)
 try:
  params=dict(n_estimators=pp['source_teacher_fixed_trees'],learning_rate=.04,num_leaves=63,min_child_samples=100,reg_lambda=10,objective='binary',n_jobs=6,random_state=20261018,verbosity=-1,deterministic=True,force_col_wise=True)
  for k,(fi,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261018).split(np.zeros(len(y)),y)):
   start_guard();p=out/f'fold{k}';p.mkdir();start=time.monotonic();assert not np.intersect1d(fi,vi).size;np.savez_compressed(p/'split.npz',train_idx=fi,valid_idx=vi)
   with threadpool_limits(6):
    tr=DistanceFeatures('base');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True));m=lgb.LGBMClassifier(**params);m.fit(X,y[fi]);lookup=OriginalLookup().fit(raw.iloc[fi].reset_index(drop=True),y[fi]);a=lookup.transform(raw.iloc[vi].reset_index(drop=True));a['orig_teacher_lgb']=m.predict_proba(V)[:,1];assert list(a)==columns and np.isfinite(a.to_numpy()).all();values[vi]=a.to_numpy();fold[vi]=k;m.booster_.save_model(str(p/'model.txt'));joblib.dump(tr,p/'transformer.joblib');joblib.dump(lookup,p/'lookup.joblib')
   row={'fold':k,'train_rows':len(fi),'valid_rows':len(vi),'model_sha256':sha(p/'model.txt'),'split_sha256':sha(p/'split.npz'),'elapsed_seconds':time.monotonic()-start};rec['folds'].append(row);dump(out/'result.json',rec);print('SOURCE_AUX',row,flush=True);del m,X,V,a;gc.collect()
  assert np.isfinite(values).all() and (fold>=0).all();np.savez_compressed(out/'features.npz',row_hash=rowhash,target=y,fold=fold,values=values,columns=np.asarray(columns));rec.update(status='complete',completed_utc=now(),features_sha256=sha(out/'features.npz'),source_rows=len(y),all_source_rows_oof_exactly_once=True);dump(out/'result.json',rec)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(out/'result.json',rec);raise
if __name__=='__main__':main()

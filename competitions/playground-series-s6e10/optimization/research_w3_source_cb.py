"""Independent source-only ordered CatBoost teacher, then fixed LGB component replacement."""
from research_w3_common import *
import argparse,time,gc,joblib
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from research_original import safe_original
from research_gpu_value_features import ValueFeatures
from research_w3_cb_values import ResourceGuard
from research_features import DistanceFeatures
from research_external_ablation import external
from research_w3_replacement import old_component,W

def cb_frame(frame):
 out=frame.copy();cats=out.select_dtypes(include=['category','object','string']).columns.tolist()
 for c in cats:out[c]=out[c].astype('object').fillna('__MISSING__').astype(str)
 return out,cats

def source():
 start_guard();pp=json.loads((ROOT/'research_w3_source_cb_plan.json').read_text());out=ROOT/'research_artifacts/w3_source_cb';out.mkdir(exist_ok=False);rec={'status':'loading','started_utc':now(),'runner_sha256':sha(__file__),'plan_sha256':sha(ROOT/'research_w3_source_cb_plan.json'),'synthetic_targets_used_in_fit_or_selection':False};dump(out/'result.json',rec);start=time.monotonic()
 try:
  original,train,test,cols,audit=safe_original();y=original.satisfaction.astype('int8').to_numpy();raw=original[cols];z=np.load(ROOT/'research_artifacts/original_teacher_v1/original_selection_split.npz');fi,vi=z['train'],z['valid'];assert len(np.intersect1d(fi,vi))==0 and len(fi)+len(vi)==len(y);rec['source_audit']=audit;params=pp['source_params']
  with threadpool_limits(6):
   tr=ValueFeatures('raw');X,cats=cb_frame(tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],pd.DataFrame(index=range(len(fi)))));V,vcats=cb_frame(tr.transform(raw.iloc[vi].reset_index(drop=True),pd.DataFrame(index=range(len(vi)))));assert cats==vcats and len(cats)==21;guard=ResourceGuard(out);m=CatBoostClassifier(**params);rec.update(status='source_selection',category_columns=cats);dump(out/'result.json',rec);m.fit(X,y[fi],cat_features=cats,eval_set=(V,y[vi]),early_stopping_rounds=150,use_best_model=True,callbacks=[guard]);assert not guard.stopped;selected=int(m.tree_count_);selection_auc=float(roc_auc_score(y[vi],m.predict_proba(V,thread_count=6)[:,1]));m.save_model(str(out/'source_selection_model.cbm'));rec.update(source_selection_auc=selection_auc,selected_trees=selected,status='source_full_fit');dump(out/'result.json',rec);del m,X,V,tr;gc.collect();start_guard()
   tr=ValueFeatures('raw');X,cats=cb_frame(tr.fit_transform(raw,y,pd.DataFrame(index=range(len(y)))));joblib.dump(tr,out/'transformer.joblib');params=dict(params,iterations=selected);m=CatBoostClassifier(**params);guard=ResourceGuard(out);m.fit(X,y,cat_features=cats,callbacks=[guard]);assert not guard.stopped and m.tree_count_==selected and np.array_equal(m.classes_,[0,1]);m.save_model(str(out/'model.cbm'));loaded=CatBoostClassifier();loaded.load_model(str(out/'model.cbm'));reload_diff=0.
   for name,df in [('train',train),('test',test)]:
    X,c2=cb_frame(tr.transform(df[cols],pd.DataFrame(index=range(len(df)))));assert cats==c2;p=m.predict_proba(X,thread_count=6)[:,1];q=loaded.predict_proba(X,thread_count=6)[:,1];reload_diff=max(reload_diff,float(np.max(np.abs(p-q))));assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),columns=np.asarray(['orig_teacher_cb'],dtype=str),values=p[:,None])
   assert reload_diff<1e-12
  rec.update(status='complete',completed_utc=now(),model_sha256=sha(out/'model.cbm'),transformer_sha256=sha(out/'transformer.joblib'),train_features_sha256=sha(out/'train_features.npz'),test_features_sha256=sha(out/'test_features.npz'),model_reload_max_abs_diff=reload_diff,elapsed_seconds=time.monotonic()-start);dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(out/'result.json',rec);raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['source','primary','confirm','oof'],required=True);a=ap.parse_args();start_guard()
 if a.split=='source':source();return
 from lightgbm import LGBMClassifier,Booster
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_source_cb_tree_{prev}/result.json').read_text())['eligible_for_next']
 df,raw,y,splits,ref,old,fold=load_split(a.split);ext,_=external('both',df.id);src=ROOT/'research_artifacts/w3_source_cb';sr=json.loads((src/'result.json').read_text());assert sr['status']=='complete' and sr['synthetic_targets_used_in_fit_or_selection'] is False and sha(src/'model.cbm')==sr['model_sha256'] and sha(src/'train_features.npz')==sr['train_features_sha256'];z=np.load(src/'train_features.npz');assert np.array_equal(z['id'],df.id);values=z['values'][:,0];out=ROOT/f'research_artifacts/w3_source_cb_tree_{a.split}';out.mkdir(exist_ok=False);params=json.loads((ROOT/'research_w3_tree_features_plan.json').read_text())['params'];allpred=np.full(len(y),np.nan);old_all=np.full(len(y),np.nan);comps=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();r=ref[vi] if a.split=='oof' else ref;nn=old[vi] if a.split=='oof' else old;base,diff=old_component(a.split,vi,df,fold,nn,r);rec={'status':'preprocessing','started_utc':now(),'plan_sha256':sha(ROOT/'research_w3_source_cb_plan.json'),'runner_sha256':sha(__file__),'source_model_sha256':sr['model_sha256'],'source_feature_sha256':sr['train_features_sha256'],'train_rows':len(fi),'valid_rows':len(vi),'old_component_reconstruction_max_abs_diff':diff};dump(run/'result.json',rec);start=time.monotonic()
  try:
   with threadpool_limits(6):
    tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
    for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
    X['orig_teacher_cb']=values[fi];V['orig_teacher_cb']=values[vi];rec.update(status='training',params=params,feature_width=X.shape[1]);dump(run/'result.json',rec);m=LGBMClassifier(**params);m.fit(X,y[fi]);assert np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(V)[:,1];m.booster_.save_model(str(run/'model.txt'));loaded=Booster(model_file=str(run/'model.txt'));rd=float(np.max(np.abs(pred-loaded.predict(V,num_threads=6))));assert rd<1e-12
   candidate=r+W*(pred-base);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();c=compare(y[vi],r,candidate);comps.append(c);allpred[vi]=pred;old_all[vi]=base;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r,old_lgb=base);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-start,vs_v6=c,model_reload_max_abs_diff=rd,model_sha256=sha(run/'model.txt'));dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del m,loaded,X,V;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=ref+W*(pred-old_all[vi]);c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,old_lgb=old_all[vi],**({'fold':fold} if fold is not None else {}));res={'status':'complete','split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

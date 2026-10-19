"""Strict inner-crossfit synthetic residual kernels added to the fixed LGB component."""
from research_w3_common import *
import argparse,time,gc,joblib,psutil
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from lightgbm import LGBMClassifier,Booster
from research_w3_residual_kernel_features import ResidualKernel,NAMES
from research_external_ablation import external
from research_features import DistanceFeatures
from research_w3_replacement import old_component,W
def guard():
 rss=psutil.Process().memory_info().rss;avail=psutil.virtual_memory().available;assert rss<=3*1024**3 and avail>=2*1024**3,'Residual kernel RAM guard';return rss,avail
def build(training_raw,training_y,training_teacher,training_idx,query_raw,query_idx,out,rec):
 """No query labels are accepted by this interface."""
 started=time.monotonic();x=np.full((len(training_idx),4),np.nan);folds=np.full(len(training_idx),-1,dtype='int8')
 def progress(stage):
  def update(g,total):
   rss,avail=guard();rec.update(stage=stage,groups_complete=g,groups_total=total,rss_bytes=rss,available_ram_bytes=avail,last_progress_utc=now());dump(out/'result.json',rec)
   if g%1000==0:print(stage,g,'/',total,now(),flush=True)
  return update
 for j,(a,b) in enumerate(KFold(5,shuffle=True,random_state=20261018).split(training_raw)):
  folds[b]=j;cache=out/f'inner_features{j}.npz'
  if cache.exists():z=np.load(cache);assert np.array_equal(z['idx'],training_idx[b]);x[b]=z['values'];continue
  model=ResidualKernel(training_raw.iloc[a],training_y[a],training_teacher[a],training_idx[a]);values=model.transform(training_raw.iloc[b],training_idx[b],progress(f'inner{j}'));x[b]=values;np.savez_compressed(cache,idx=training_idx[b],values=values);del model;gc.collect();guard()
 assert np.isfinite(x).all() and (folds>=0).all();np.savez_compressed(out/'training_features.npz',idx=training_idx,inner_fold=folds,values=x)
 model=ResidualKernel(training_raw,training_y,training_teacher,training_idx);v=model.transform(query_raw,query_idx,progress('outer_validation'));joblib.dump(model,out/'residual_kernel.joblib');del model;gc.collect();loaded=joblib.load(out/'residual_kernel.joblib');n=min(1024,len(query_idx));again=loaded.transform(query_raw.iloc[:n],query_idx[:n]);diff=float(np.max(np.abs(again-v[:n])));assert diff<1e-12;del loaded;gc.collect();guard();np.savez_compressed(out/'query_features.npz',idx=query_idx,values=v);rec.update(kernel_feature_seconds=time.monotonic()-started,kernel_reload_max_abs_diff=diff,kernel_model_sha256=sha(out/'residual_kernel.joblib'),inner_feature_own_label_overlap=0,outer_query_source_overlap=0,inner_split_uses_labels=False);dump(out/'result.json',rec);return x,v
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard()
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_residual_kernel_{prev}/result.json').read_text())['eligible_for_next']
 df,raw,y,splits,ref,old,fold=load_split(a.split);ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();out=ROOT/f'research_artifacts/w3_residual_kernel_{a.split}';out.mkdir(exist_ok=False);pp=json.loads((ROOT/'research_w3_residual_kernel_plan.json').read_text());allpred=np.full(len(y),np.nan);old_all=np.full(len(y),np.nan);comps=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();r=ref[vi] if a.split=='oof' else ref;nn=old[vi] if a.split=='oof' else old;base,diff=old_component(a.split,vi,df,fold,nn,r);rec={'status':'crossfitting_neighborhood_features','started_utc':now(),'plan_sha256':sha(ROOT/'research_w3_residual_kernel_plan.json'),'runner_sha256':sha(__file__),'kernel_code_sha256':sha(ROOT/'research_w3_residual_kernel_features.py'),'source_teacher_model_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/model.txt'),'source_feature_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'train_rows':len(fi),'valid_rows':len(vi),'old_component_reconstruction_max_abs_diff':diff};dump(run/'result.json',rec);started=time.monotonic()
  try:
   with threadpool_limits(6):
    kt,kv=build(raw.iloc[fi].reset_index(drop=True),y[fi],teacher[fi],fi,raw.iloc[vi].reset_index(drop=True),vi,run,rec);start_guard();tr=DistanceFeatures('distance_stats_numeric');x=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);v=tr.transform_te(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
    for c in ext:x[c]=ext[c].to_numpy()[fi];v[c]=ext[c].to_numpy()[vi]
    for j,c in enumerate(NAMES):x[c]=kt[:,j];v[c]=kv[:,j]
    guard();rec.update(status='training_lgb',feature_width=x.shape[1],params=pp['params']);dump(run/'result.json',rec);m=LGBMClassifier(**pp['params']);m.fit(x,y[fi]);assert np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(v)[:,1];m.booster_.save_model(str(run/'model.txt'));loaded=Booster(model_file=str(run/'model.txt'));rd=float(np.max(np.abs(pred-loaded.predict(v,num_threads=6))));assert rd<1e-12
   candidate=r+W*(pred-base);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();c=compare(y[vi],r,candidate);comps.append(c);allpred[vi]=pred;old_all[vi]=base;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r,old_lgb=base);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-started,vs_v6=c,model_reload_max_abs_diff=rd,model_sha256=sha(run/'model.txt'));dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del m,loaded,x,v;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps,'completed_utc':now()});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=ref+W*(pred-old_all[vi]);c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,old_lgb=old_all[vi],**({'fold':fold} if fold is not None else {}));rec={'status':'complete','split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()

"""A bounded, fixed ExtraTrees family screen with training-only preprocessing."""
from research_w3_common import *
import argparse,time,gc,joblib,psutil
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from research_features import DistanceFeatures
from research_external_ablation import external
class ETFeatures:
 def fit_transform(self,raw,y,ext):
  self.base=DistanceFeatures('distance_stats_numeric');x=self.base.fit_transform_te(raw,y);self.ext_cols=list(ext)
  for c in ext:x[c]=ext[c].to_numpy()
  self.cats=x.select_dtypes(include=['category','object','string']).columns.tolist();self.num=[c for c in x if c not in self.cats];self.medians=x[self.num].median();self.ohe=OneHotEncoder(sparse_output=False,dtype=np.float32,handle_unknown='ignore');self.ohe.fit(x[self.cats].astype(str));return self.finish(x)
 def finish(self,x):
  a=np.concatenate([x[self.num].fillna(self.medians).to_numpy('float32'),self.ohe.transform(x[self.cats].astype(str))],axis=1);assert np.isfinite(a).all();return a
 def transform(self,raw,ext):
  assert list(ext)==self.ext_cols;x=self.base.transform_te(raw)
  for c in ext:x[c]=ext[c].to_numpy()
  return self.finish(x)
def guard():
 rss=psutil.Process().memory_info().rss;avail=psutil.virtual_memory().available;assert rss<=3*1024**3 and avail>=2*1024**3,'ExtraTrees RAM guard';return rss,avail
def fit(m,x,y,out,rec,target):
 built=len(m.estimators_) if hasattr(m,'estimators_') else 0
 while built<target:
  m.set_params(n_estimators=min(target,built+32));start=time.monotonic();m.fit(x,y);built=len(m.estimators_);rss,avail=guard();rec.update(status='training',completed_trees=built,last_chunk_seconds=time.monotonic()-start,rss_bytes=rss,available_ram_bytes=avail,last_progress_utc=now());dump(out/'result.json',rec);print(out.name,{k:rec[k] for k in ['completed_trees','last_chunk_seconds','rss_bytes','last_progress_utc']},flush=True)
  if built%128==0 or built==target:joblib.dump(m,out/'checkpoint.joblib');rec['checkpoint_trees']=built;dump(out/'result.json',rec)
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();pp=json.loads((ROOT/'research_w3_extra_trees_plan.json').read_text());df,raw,y,splits,ref,_,fold=load_split('primary' if a.split=='probe' else a.split);ext,_=external('both',df.id);out=ROOT/f'research_artifacts/w3_extra_trees_{a.split}';out.mkdir(exist_ok=False);rec={'status':'preprocessing','started_utc':now(),'plan_sha256':sha(ROOT/'research_w3_extra_trees_plan.json'),'runner_sha256':sha(__file__),'split':a.split};dump(out/'result.json',rec);comps=[];allpred=np.full(len(y),np.nan)
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_extra_trees_{prev}/result.json').read_text())['eligible_for_next']
 with threadpool_limits(6):
  for k,(fi,vi) in enumerate(splits):
   start_guard();run=out/f'fold{k}';run.mkdir();rec.update(fold=k,train_rows=len(fi),valid_rows=len(vi));params=pp['params'];probe=ROOT/'research_artifacts/w3_extra_trees_probe/fold0'
   if a.split=='primary':
    pr=json.loads((probe/'result.json').read_text());assert pr['status']=='complete' and pr['plan_sha256']==rec['plan_sha256'];z=np.load(probe/'training_identity.npz');assert np.array_equal(z['idx'],fi) and np.array_equal(z['id'],df.id.iloc[fi]) and np.array_equal(z['target'],y[fi]);assert sha(probe/'transformer.joblib')==pr['transformer_sha256'] and sha(probe/'checkpoint.joblib')==pr['model_sha256'];tr=joblib.load(probe/'transformer.joblib');x=tr.transform(raw.iloc[fi].reset_index(drop=True),ext.iloc[fi].reset_index(drop=True));m=joblib.load(probe/'checkpoint.joblib');assert len(m.estimators_)==64
   else:tr=ETFeatures();x=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));m=ExtraTreesClassifier(**params)
   joblib.dump(tr,run/'transformer.joblib');rec['feature_width']=x.shape[1];start=time.monotonic();m=fit(m,x,y[fi],run,rec,64 if a.split=='probe' else 512);seconds=time.monotonic()-start
   if a.split=='probe':
    rss,avail=guard();bound=rss+(512-64)*8191*104;projected=seconds*8;rec.update(status='complete',score_evaluated=False,outer_validation_labels_used=False,fit64_seconds=seconds,projected_fit512_seconds=projected,conservative_final_rss_bound_bytes=bound,transformer_sha256=sha(run/'transformer.joblib'),model_sha256=sha(run/'checkpoint.joblib'),completed_utc=now());np.savez_compressed(run/'training_identity.npz',idx=fi,id=df.id.iloc[fi].to_numpy(),target=y[fi]);assert projected<=1200 and bound<=3*1024**3,'ExtraTrees projected resources exceed guard';dump(run/'result.json',rec);dump(out/'result.json',rec);dump(ROOT/'research_w3_extra_trees_hardware.json',rec);return
   v=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));assert np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(v)[:,1];loaded=joblib.load(run/'checkpoint.joblib');rd=float(np.max(np.abs(pred-loaded.predict_proba(v)[:,1])));assert rd<1e-12;r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);allpred[vi]=pred;comps.append(c);np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r);rec.update(status='complete',vs_v6=c,model_reload_max_abs_diff=rd,model_sha256=sha(run/'checkpoint.joblib'),transformer_sha256=sha(run/'transformer.joblib'),completed_utc=now());dump(run/'result.json',rec);del m,loaded,x,v;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps,'completed_utc':now()});return
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,**({'fold':fold} if fold is not None else {}));rec={'status':'complete','split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_w3_extra_trees_last_failure.json',{'status':'failed','ended_utc':now(),'argv':os.sys.argv,'error':repr(e)});raise

"""One fixed Ordered CatBoost value view, bounded CPU/RAM and unchanged acceptance gates."""
from research_w3_common import *
import argparse,time,gc,joblib,psutil
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_external_ablation import external
from research_w3_cb_features import CBValueFeatures
class ResourceGuard:
 def __init__(self,out):self.out=out;self.rows=[];self.stopped=False;self.proc=psutil.Process()
 def after_iteration(self,info):
  if info.iteration%50==0:
   rss=self.proc.memory_info().rss;available=psutil.virtual_memory().available;self.rows.append({'iteration':info.iteration,'rss_bytes':rss,'available_ram_bytes':available});dump(self.out/'memory_guard.json',{'samples':self.rows,'stopped':self.stopped})
   if rss>3*1024**3 or available<2*1024**3:self.stopped=True;return False
  return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();pp=json.loads((ROOT/'research_w3_cb_values_plan.json').read_text());df,raw,y,splits,ref,old,fold=load_split(a.split);ext,_=external('both',df.id);out=ROOT/'research_artifacts'/f'w3_cb_values_{a.split}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);comps=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'preprocessing','started_utc':now(),'params':pp['params'],'plan_sha256':sha(ROOT/'research_w3_cb_values_plan.json'),'runner_sha256':sha(__file__),'feature_code_sha256':sha(ROOT/'research_w3_cb_features.py'),'train_rows':len(fi),'valid_rows':len(vi),'outer_validation_used_in_fit':False};dump(run/'result.json',rec);start=time.monotonic()
  try:
   with threadpool_limits(6):
    tr=CBValueFeatures();X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));assert len(tr.catcols)==21;joblib.dump(tr,run/'transformer.joblib');rec.update(status='training',feature_width=X.shape[1],category_columns=tr.catcols);dump(run/'result.json',rec);guard=ResourceGuard(run);m=CatBoostClassifier(**pp['params']);m.fit(X,y[fi],cat_features=tr.catcols,callbacks=[guard]);assert not guard.stopped and m.tree_count_==pp['params']['iterations'] and np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(V,thread_count=6)[:,1];m.save_model(str(run/'model.cbm'));loaded=CatBoostClassifier();loaded.load_model(str(run/'model.cbm'));again=loaded.predict_proba(V,thread_count=6)[:,1];diff=float(np.max(np.abs(pred-again)));assert diff<1e-12;resolved=m.get_all_params()
   r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);comps.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-start,vs_v6=c,model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.cbm'),resolved_params=resolved);dump(run/'result.json',rec);del m,loaded,X,V;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,**({'fold':fold} if fold is not None else {}));res={'status':'complete','variant':'cb_values','split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

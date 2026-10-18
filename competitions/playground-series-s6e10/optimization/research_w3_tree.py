"""Fixed-capacity tree feature tests with frozen references and stage gates."""
from research_w3_common import *
import argparse,time,joblib,gc,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_external_ablation import external
from research_original import safe_original
from research_w3_tree_features import SourceProfiles,TreeFeatures
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['digits_counts','source_profiles'],required=True);ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();pp=json.loads((ROOT/'research_w3_tree_features_plan.json').read_text());df,raw,y,splits,ref,old,fold=load_split(a.split);ext,_=external('both',df.id);profiles=None;sourceaudit=None
 if a.variant=='source_profiles':
  source,_,_,cols,sourceaudit=safe_original();profiles=SourceProfiles().fit(source[cols],source.satisfaction.astype('int8').to_numpy());del source
 out=ROOT/'research_artifacts'/f'w3_tree_{a.variant}_{a.split}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);comps=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'preprocessing','started_utc':now(),'params':pp['params'],'plan_sha256':sha(ROOT/'research_w3_tree_features_plan.json'),'feature_code_sha256':sha(ROOT/'research_w3_tree_features.py'),'runner_sha256':sha(__file__),'source_audit':sourceaudit,'train_rows':len(fi),'valid_rows':len(vi),'external_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')};dump(run/'result.json',rec);t=time.monotonic()
  try:
   with threadpool_limits(6):
    tr=TreeFeatures(a.variant,profiles);X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib');rec.update(status='training',feature_width=X.shape[1],feature_names=list(X));dump(run/'result.json',rec);m=lgb.LGBMClassifier(**pp['params']);m.fit(X,y[fi]);pred=m.predict_proba(V)[:,1];m.booster_.save_model(str(run/'model.txt'));loaded=lgb.Booster(model_file=str(run/'model.txt'));again=loaded.predict(V,num_threads=6);diff=float(np.max(np.abs(pred-again)));assert diff<1e-12
   assert np.isfinite(pred).all();r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);comps.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-t,vs_v6=c,model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.txt'));dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del m,loaded,X,V;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,**({'fold':fold} if fold is not None else {}));result={'status':'complete','variant':a.variant,'split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

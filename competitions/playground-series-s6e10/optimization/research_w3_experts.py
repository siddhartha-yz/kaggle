"""Two fixed semantic experts; outer validation never selects groups or model parameters."""
from research_w3_common import *
import argparse,time,gc,joblib,lightgbm as lgb
from threadpoolctl import threadpool_limits
from research_features import DistanceFeatures
from research_external_ablation import external
from research_w3_replacement import old_component,W
def group(raw,variant):return raw['Class'].eq('Business').to_numpy() if variant=='class' else raw['Type of Travel'].eq('Business travel').to_numpy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['class','travel'],required=True);ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();df,raw,y,splits,ref,old,fold=load_split(a.split);ext,_=external('both',df.id);out=ROOT/'research_artifacts'/f'w3_experts_{a.variant}_{a.split}';out.mkdir(exist_ok=False);params=json.loads((ROOT/'research_w3_tree_features_plan.json').read_text())['params'];allpred=np.full(len(y),np.nan);old_all=np.full(len(y),np.nan);comps=[];groups=group(raw,a.variant)
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();r=ref[vi] if a.split=='oof' else ref;nn=old[vi] if a.split=='oof' else old;base,diff=old_component(a.split,vi,df,fold,nn,r);rec={'status':'preprocessing','started_utc':now(),'variant':a.variant,'params':params,'plan_sha256':sha(ROOT/'research_w3_experts_plan.json'),'runner_sha256':sha(__file__),'train_rows':len(fi),'valid_rows':len(vi),'old_component_reconstruction_max_abs_diff':diff,'experts':[]};dump(run/'result.json',rec);start=time.monotonic()
  try:
   with threadpool_limits(6):
    tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
    for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
    pred=np.full(len(vi),np.nan)
    for g in [False,True]:
     start_guard();mi=groups[fi]==g;mv=groups[vi]==g;assert mi.sum()>1000 and np.unique(y[fi][mi]).size==2;m=lgb.LGBMClassifier(**params);m.fit(X.loc[mi],y[fi][mi]);p=m.predict_proba(V.loc[mv])[:,1];file=run/f'expert{int(g)}.txt';m.booster_.save_model(str(file));loaded=lgb.Booster(model_file=str(file));again=loaded.predict(V.loc[mv],num_threads=6);rd=float(np.max(np.abs(p-again)));assert rd<1e-12;pred[mv]=p;rec['experts'].append({'group':bool(g),'train_rows':int(mi.sum()),'valid_rows':int(mv.sum()),'model_sha256':sha(file),'model_reload_max_abs_diff':rd});dump(run/'result.json',rec);print('EXPERT',a.variant,a.split,k,bool(g),rec['experts'][-1],flush=True);del m,loaded;gc.collect()
   assert np.isfinite(pred).all();candidate=r+W*(pred-base);assert ((candidate>=0)&(candidate<=1)).all();c=compare(y[vi],r,candidate);comps.append(c);allpred[vi]=pred;old_all[vi]=base;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r,old_lgb=base);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-start,vs_v6=c);dump(run/'result.json',rec);del X,V;gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=ref+W*(pred-old_all[vi]);c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,old_lgb=old_all[vi],**({'fold':fold} if fold is not None else {}));res={'status':'complete','variant':a.variant,'split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

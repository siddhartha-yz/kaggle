"""Original-covariate-only rating relation selection; frozen complete paired fivefold run."""
from research_r4_common import *
import itertools,time,gc,joblib
import pandas as pd,lightgbm as lgb
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from research_original import safe_original
from research_features import DistanceFeatures
from research_external_ablation import external
from research_oof import params
from research_w3_replacement import old_component,W
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']

def select_pairs(raw):
 assert 'satisfaction' not in raw and 'id' not in raw
 stats=[]
 for a,b in itertools.combinations(sorted(RATINGS),2):
  u=raw[a].to_numpy();v=raw[b].to_numpy();assert np.isfinite(u).all() and np.isfinite(v).all()
  eq=float(np.mean(u==v));pu=pd.Series(u).value_counts(normalize=True);pv=pd.Series(v).value_counts(normalize=True);expected=float(sum(pu.get(i,0)*pv.get(i,0) for i in range(6)))
  stats.append({'columns':[a,b],'normalized_mi':float(normalized_mutual_info_score(u,v)),'equality_rate':eq,'marginal_equality_expectation':expected})
 stats.sort(key=lambda r:(-r['normalized_mi'],r['columns']))
 return stats[:12],stats

class RelationFeatures:
 def __init__(self,pairs):self.pairs=pairs
 def fit_transform(self,raw,y,ext):
  self.base=DistanceFeatures('distance_stats_numeric');return self.finish(raw,self.base.fit_transform_te(raw,y),ext)
 def transform(self,raw,ext):return self.finish(raw,self.base.transform_te(raw),ext)
 def finish(self,raw,out,ext):
  out=out.copy()
  for c in ext:out[c]=ext[c].to_numpy()
  for j,(a,b) in enumerate(self.pairs):
   u=raw[a].to_numpy();v=raw[b].to_numpy();out[f'relation_eq_{j:02}']=(u==v).astype('int8');out[f'relation_absdiff_{j:02}']=np.abs(u-v).astype('int8')
  return out

def main():
 guard();out=ROOT/'research_artifacts/r4_rating_relations_oof';out.mkdir(exist_ok=True)
 selection=out/'selection.json'
 if not selection.exists():
  source,_,_,cols,audit=safe_original();chosen,allpairs=select_pairs(source[cols]);dump(selection,{'frozen_utc':now(),'chosen':chosen,'all_pairs':allpairs,'source':audit,'target_used_for_pair_selection':False,'selector_code_sha256':sha(__file__),'plan_sha256':sha(ROOT/'research_r4_plan.json')});del source;gc.collect()
 selected=json.loads(selection.read_text());assert selected['selector_code_sha256']==sha(__file__)
 pairs=[r['columns'] for r in selected['chosen']];df,raw,y,splits,ref,old,fold=load_split('oof');ext,_=external('both',df.id);predall=np.full(len(y),np.nan);oldall=np.full(len(y),np.nan)
 dump(ROOT/'research_r4_state.json',{'updated_utc':now(),'status':'training','route':'rating_relations','active_training':True,'pid':os.getpid(),'selection_sha256':sha(selection),'plan':'research_r4_plan.json'})
 for k,(fi,vi) in enumerate(splits):
  guard();run=out/f'fold{k}';run.mkdir(exist_ok=True);result=run/'result.json'
  if result.exists() and json.loads(result.read_text()).get('status')=='complete':
   z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);predall[vi]=z['pred'];oldall[vi]=z['old_lgb'];continue
  tick=time.monotonic();r=ref[vi];oldlgb,err=old_component('oof',vi,df,fold,old[vi],r)
  rec={'status':'preprocessing','started_utc':now(),'fold':k,'train_rows':len(fi),'valid_rows':len(vi),'params':params('stats639'),'selection_sha256':sha(selection),'runner_sha256':sha(__file__),'plan_sha256':sha(ROOT/'research_r4_plan.json'),'reference_reconstruction_error':err};dump(result,rec);print('START',k,flush=True)
  try:
   with threadpool_limits(6):
    tr=RelationFeatures(pairs);X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True));V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));rec.update(status='training',feature_width=X.shape[1]);dump(result,rec)
    model=lgb.LGBMClassifier(**params('stats639'));model.fit(X,y[fi]);pred=model.predict_proba(V)[:,1];model.booster_.save_model(str(run/'model.txt'));joblib.dump(tr,run/'transformer.joblib');reload=lgb.Booster(model_file=str(run/'model.txt')).predict(V,num_threads=6);reloaderr=float(np.max(np.abs(pred-reload)));assert reloaderr<1e-12
   cand=r+W*(pred-oldlgb);assert np.isfinite(cand).all() and (cand>=0).all() and (cand<=1).all();predall[vi]=pred;oldall[vi]=oldlgb
   np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,old_lgb=oldlgb,candidate=cand)
   rec.update(status='complete',ended_utc=now(),elapsed_seconds=time.monotonic()-tick,model_sha256=sha(run/'model.txt'),prediction_sha256=sha(run/'validation.npz'),reload_error=reloaderr,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,metric_policy='All five folds complete before scoring; no interim outcome-based stopping.');dump(result,rec);print('DONE',k,rec['elapsed_seconds'],flush=True);del X,V,model,tr;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(result,rec);raise
 assert np.isfinite(predall).all();cand=ref+W*(predall-oldall);np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,candidate=cand,v6=ref,old_lgb=oldall)
 result=evidence(y,ref,cand,fold);result.update(status='complete',ended_utc=now(),route='rating_relations',selection_sha256=sha(selection),oof_sha256=sha(out/'oof.npz'));dump(out/'result.json',result);dump(ROOT/'research_r4_state.json',{'updated_utc':now(),'status':'route_complete','route':'rating_relations','active_training':False,'result':str(out/'result.json'),'eligible_for_replication':result['eligible_for_new_partition_replication']});print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

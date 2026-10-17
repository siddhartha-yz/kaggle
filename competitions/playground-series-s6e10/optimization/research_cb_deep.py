"""One deeper CatBoost configuration; frozen core/weight and strict existing splits."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,json,time,hashlib,argparse,signal,gc
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from research_features import DistanceFeatures
from research_external_ablation import external
from research_nested_stack import outer_predictions
from compare_validation import compare
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.now(timezone.utc)
def core_from_columns(pred):return .625*pred[:,0]+.125*pred[:,1]+.25*pred[:,2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','confirm','oof'],required=True);p.add_argument('--trees',type=int);p.add_argument('--weight',type=float);a=p.parse_args();assert a.split=='primary' or (a.trees is not None and a.weight in [.25,1.])
 assert json.loads((ROOT/'research_stage11_queue_state.json').read_text())['status']=='complete'
 protocol=json.loads((ROOT/'research_protocol.json').read_text());frozen=json.loads((ROOT/'research_cb_deep_plan.json').read_text());deadline=datetime.fromisoformat(protocol['training_hard_cutoff_utc'])
 if a.split=='primary':deadline=min(deadline,now()+timedelta(seconds=frozen['primary_time_cap_seconds']))
 def stop(s,f):raise TimeoutError('Training time limit')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((deadline-now()).total_seconds())))
 assert frozen['core_weights']=={'lgb_original':.625,'residual_prior05':.25,'cb_original':.125}
 params=dict(frozen['cb_params']);params['iterations']=a.trees if a.trees is not None else params['iterations'];out=ROOT/'research_artifacts'/f'cb_deep_{a.split}';out.mkdir(exist_ok=True)
 plan={'split':a.split,'params':params,'weight':a.weight,'early_stopping':a.split=='primary','frozen_plan_sha256':sha(ROOT/'research_cb_deep_plan.json'),'preprocessing_sha256':sha(ROOT/'research_features.py'),'source_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'runner_sha256':sha(__file__),'created_utc':now().isoformat()}
 if (out/'plan.json').exists():
  old=json.loads((out/'plan.json').read_text());assert all(old[k]==plan[k] for k in ['params','weight','frozen_plan_sha256','preprocessing_sha256','source_features_sha256'])
 else:dump(out/'plan.json',plan)
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.astype('int8').to_numpy();raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id);fold=None
 if a.split=='oof':
  z=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fold=z['fold'];splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(5)]
 else:
  z=np.load(ROOT/('artifacts/split_seed42.npz' if a.split=='primary' else 'iteration_artifacts/split_confirm20261017.npz'));splits=[(z['train'],z['valid'])]
 for k,(fi,vi) in enumerate(splits):
  run=out/f'fold{k}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists() and json.loads((run/'result.json').read_text()).get('status')=='complete':continue
  assert now()<datetime.fromisoformat(protocol['training_start_cutoff_utc']);start=time.monotonic();rec={'status':'preprocessing','fold':k,'started_utc':now().isoformat(),'train_rows':len(fi),'valid_rows':len(vi)};dump(run/'result.json',rec)
  try:
   tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
   for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
   cats=X.select_dtypes('category').columns.tolist()
   for c in cats:X[c]=X[c].astype(str);V[c]=V[c].astype(str)
   model=CatBoostClassifier(**params);kwargs={'eval_set':(V,y[vi]),'early_stopping_rounds':150,'use_best_model':True} if a.split=='primary' else {};rec['status']='training';dump(run/'result.json',rec);model.fit(X,y[fi],cat_features=cats,**kwargs)
   pred=model.predict_proba(V,thread_count=6)[:,1];assert np.isfinite(pred).all();model.save_model(str(run/'model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(run/'model.cbm'));again=loaded.predict_proba(V,thread_count=6)[:,1];assert np.max(np.abs(pred-again))<1e-14
   selected=int(model.tree_count_);assert a.split=='primary' or selected==a.trees;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);rec.update(status='complete',selected_trees=selected,model_sha256=sha(run/'model.cbm'),reload_maxdiff=float(np.max(np.abs(pred-again))),elapsed_seconds=time.monotonic()-start,ended_utc=now().isoformat());dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del model,loaded,X,V;gc.collect()
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now().isoformat());dump(run/'result.json',rec);raise
 if a.split=='oof':
  pred=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(splits):z=np.load(out/f'fold{k}/validation.npz');assert np.array_equal(z['idx'],vi);pred[vi]=z['pred']
  assert np.isfinite(pred).all();idx=np.arange(len(y));P=outer_predictions('oof',idx,df,fold);target=y;ids=df.id.to_numpy();np.savez_compressed(out/'raw_oof.npz',id=ids,target=y,fold=fold,pred=pred)
 else:
  idx=splits[0][1];z=np.load(out/'fold0/validation.npz');pred=z['pred'];target=y[idx];ids=df.id.iloc[idx].to_numpy();P=outer_predictions(a.split,idx,df)
 base=P[:,0];core=core_from_columns(P);weights=frozen['candidate_deep_weights'] if a.split=='primary' else [a.weight];result={'status':'complete','split':a.split,'selected_trees':[json.loads((out/f'fold{k}/result.json').read_text())['selected_trees'] for k in range(len(splits))],'raw_vs_v4':compare(target,base,pred),'candidates':{},'completed_utc':now().isoformat()}
 for w in weights:
  candidate=(1-w)*core+w*pred;c=compare(target,base,candidate);cc=compare(target,core,candidate);r={'vs_v4':c,'vs_core':cc}
  if a.split=='oof':
   r['byfold']=[compare(target[fold==k],base[fold==k],candidate[fold==k]) for k in range(5)];r['eligible']=c['delta_auc']>=.0002 and c['conditional_95pct_ci'][0]>0 and all(z['delta_auc']>0 for z in r['byfold']) and cc['delta_auc']>0;np.savez_compressed(out/'candidate_oof.npz',id=ids,target=target,fold=fold,pred=candidate)
  else:np.savez_compressed(out/f'candidate_w{w}.npz',idx=idx,id=ids,target=target,pred=candidate)
  result['candidates'][str(w)]=r
 dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

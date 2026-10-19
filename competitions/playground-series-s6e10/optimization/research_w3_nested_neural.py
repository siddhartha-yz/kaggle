"""Strict nested stacking with newly informative raw-value neural base predictions."""
from research_w3_common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['XDG_CACHE_HOME']=str(ROOT.parent/'.cache-nested-neural-w3')
import argparse,time,gc,joblib,cloudpickle,psutil,resource
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier,Booster
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_gpu_value_features import ValueFeatures
from research_nested_stack import outer_predictions
from research_external_ablation import external

WEIGHTS=np.array([45,9,18,24,32],dtype='float64')/128
def pp():return json.loads((ROOT/'research_w3_nested_neural_plan.json').read_text())
def hashfile(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def guard():
 rss=psutil.Process().memory_info().rss;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;avail=psutil.virtual_memory().available;assert peak<=4*1024**3 and avail>=3*1024**3,'NestedNN CPU RAM guard';return {'rss_bytes':rss,'peak_rss_bytes':peak,'available_ram_bytes':avail}
def neural(raw,y,df,inner_train,inner_meta,out):
 out.mkdir(exist_ok=True);path=out/'result.json';cfg=pp()['nn_config'];predfile=out/'prediction.npz';train_hash=hashlib.sha256(np.asarray(inner_train,dtype='int64').tobytes()).hexdigest();target_hash=hashlib.sha256(y[inner_train].tobytes()).hexdigest()
 if path.exists() and json.loads(path.read_text()).get('status')=='complete':
  rec=json.loads(path.read_text());assert rec['config']==cfg and rec['model_sha256']==hashfile(out/'model.pkl') and rec['prediction_sha256']==hashfile(predfile) and rec['training_indices_sha256']==train_hash and rec['training_target_sha256']==target_hash;z=np.load(predfile);assert np.array_equal(z['idx'],inner_meta) and np.array_equal(z['id'],df.id.iloc[inner_meta]) and np.array_equal(z['target'],y[inner_meta]);return z['pred'],rec
 tr=ValueFeatures('raw');x=tr.fit_transform(raw.iloc[inner_train].reset_index(drop=True),y[inner_train],pd.DataFrame(index=range(len(inner_train))));v=tr.transform(raw.iloc[inner_meta].reset_index(drop=True),pd.DataFrame(index=range(len(inner_meta))));joblib.dump(tr,out/'transformer.joblib');rec={'status':'training','started_utc':now(),'config':cfg,'train_rows':len(inner_train),'meta_rows':len(inner_meta),'training_indices_sha256':train_hash,'training_target_sha256':target_hash,'no_outer_validation_labels_or_features_in_fit':True,'epochs':[]};dump(path,rec);model=RealMLP_TD_Classifier(**cfg);original=TabNNModule.on_train_epoch_end;last=[time.monotonic()];started=time.monotonic()
 def epoch_hook(module):
  original(module);t=time.monotonic();row={'epoch':int(module.current_epoch)+1,'seconds':t-last[0],**guard()};last[0]=t;rec['epochs'].append(row);dump(path,rec);print('CPU innerNN',row,flush=True);assert t-started<=600,'Nested NN fit exceeded600seconds'
 TabNNModule.on_train_epoch_end=epoch_hook
 try:model.fit(x,y[inner_train])
 finally:TabNNModule.on_train_epoch_end=original
 assert len(rec['epochs'])==3 and not model.cv_alg_interface_.model.creator.is_cv and np.array_equal(model.classes_,[0,1]);assert all(p.device.type=='cpu' for p in model.cv_alg_interface_.model.parameters());fitseconds=time.monotonic()-started;pred=model.predict_proba(v)[:,1];assert np.isfinite(pred).all();model.to('cpu')
 with open(out/'model.pkl','wb') as f:cloudpickle.dump(model,f)
 del model,x;gc.collect()
 with open(out/'model.pkl','rb') as f:loaded=cloudpickle.load(f)
 again=loaded.predict_proba(v.iloc[:2048])[:,1];diff=float(np.max(np.abs(pred[:2048]-again)));assert diff<1e-7;del loaded,v;gc.collect();np.savez_compressed(predfile,idx=inner_meta,id=df.id.iloc[inner_meta].to_numpy(),target=y[inner_meta],pred=pred);rec.update(status='complete',completed_utc=now(),fit_seconds=fitseconds,model_reload_max_abs_diff=diff,model_sha256=hashfile(out/'model.pkl'),transformer_sha256=hashfile(out/'transformer.joblib'),prediction_sha256=hashfile(predfile),gpu_hidden_from_worker=True,**guard());dump(path,rec);return pred,rec
def deep_cb(raw,y,ext,teacher,inner_train,inner_meta,cache,out):
 path=out/'cb10_inner_prediction.npz';modelpath=out/'cb10_inner.cbm'
 if path.exists():
  z=np.load(path);assert np.array_equal(z['idx'],inner_meta) and np.array_equal(z['target'],y[inner_meta]);assert modelpath.exists();return z['pred'],{'reused':True,'model_sha256':hashfile(modelpath)}
 tr=joblib.load(cache/'inner_transformer.joblib');x=tr.transform_te(raw.iloc[inner_train].reset_index(drop=True));v=tr.transform_te(raw.iloc[inner_meta].reset_index(drop=True))
 for c in ext:x[c]=ext[c].to_numpy()[inner_train];v[c]=ext[c].to_numpy()[inner_meta]
 cats=x.select_dtypes('category').columns.tolist()
 for c in cats:x[c]=x[c].astype(str);v[c]=v[c].astype(str)
 params=pp()['cb10_params'];m=CatBoostClassifier(**params);start=time.monotonic();m.fit(x,y[inner_train],cat_features=cats);assert m.tree_count_==813 and np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(v,thread_count=6)[:,1];m.save_model(str(modelpath));loaded=CatBoostClassifier();loaded.load_model(str(modelpath));diff=float(np.max(np.abs(pred-loaded.predict_proba(v,thread_count=6)[:,1])));assert diff<1e-12;np.savez_compressed(path,idx=inner_meta,target=y[inner_meta],pred=pred);rec={'model_sha256':hashfile(modelpath),'model_reload_max_abs_diff':diff,'fit_seconds':time.monotonic()-start,**guard()};del m,loaded,x,v;gc.collect();return pred,rec
def cached_inner(cache,raw,y,ext,teacher,inner_meta):
 rec=json.loads((cache/'result.json').read_text());assert rec['status']=='complete';tr=joblib.load(cache/'inner_transformer.joblib');v=tr.transform_te(raw.iloc[inner_meta[:2048]].reset_index(drop=True))
 for c in ext:v[c]=ext[c].to_numpy()[inner_meta[:2048]]
 arrays=[];audits={}
 for name in ['lgb_original','cb_original','residual_prior05']:
  file=cache/(name+('.cbm' if name=='cb_original' else '.txt'));assert hashfile(file)==rec['model_hashes'][name];z=np.load(cache/f'{name}_inner_prediction.npz');assert np.array_equal(z['idx'],inner_meta) and np.array_equal(z['target'],y[inner_meta]);pred=z['pred'];arrays.append(pred)
  if name=='cb_original':
   m=CatBoostClassifier();m.load_model(str(file));cv=v.copy()
   for c in cv.select_dtypes('category'):cv[c]=cv[c].astype(str)
   actual=m.predict_proba(cv,thread_count=6)[:,1];del cv
  else:
   m=Booster(model_file=str(file));actual=m.predict(v,raw_score=name=='residual_prior05',num_threads=6)
   if name=='residual_prior05':actual=expit(actual+.5*logit(np.clip(teacher[inner_meta[:2048]],1e-6,1-1e-6)))
  diff=float(np.max(np.abs(actual-pred[:2048])));assert diff<1e-12;audits[name]={'model_sha256':rec['model_hashes'][name],'cached_prediction_reload_max_abs_diff':diff};del m
 return arrays,audits
def features(p,teacher,residual=False):
 z=np.column_stack([logit(np.clip(p,1e-6,1-1e-6)),logit(np.clip(teacher,1e-6,1-1e-6))]);base=logit(np.clip(p@WEIGHTS,1e-6,1-1e-6));return np.column_stack([z,z-base[:,None]]) if residual else z
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);a=ap.parse_args();split='primary' if a.split=='probe' else a.split;start_guard();torch.set_num_threads(6);torch.set_num_interop_threads(1);frozen=pp();selected=None
 if split!='primary':
  selection=json.loads((ROOT/'research_w3_nested_neural_selection.json').read_text());assert selection['eligible'] and selection['plan_sha256']==sha(ROOT/'research_w3_nested_neural_plan.json');selected=selection['method']
  if split=='oof':assert json.loads((ROOT/'research_artifacts/w3_nested_neural_confirm/result.json').read_text())['eligible_for_next']
 assert hashfile(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36';df,raw,y,splits,ref,old,fold=load_split(split);ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();out=ROOT/f'research_artifacts/w3_nested_neural_{split}';out.mkdir(exist_ok=True);methods=[selected] if selected else ['logistic','residual_tree'];allpred={m:np.full(len(y),np.nan) for m in methods};comparisons=[]
 cacheplan=json.loads((ROOT/f'research_artifacts/nested_stack_{split}/plan.json').read_text());assert cacheplan['preprocessing_sha256']==sha(ROOT/'research_features.py') and cacheplan['source_features_sha256']==sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')
 for k,(fi,vi) in enumerate(splits):
  start_guard();cache=ROOT/f'research_artifacts/nested_stack_{split}/fold{k}';indices=np.load(cache/'nested_indices.npz');assert np.array_equal(indices['outer_train'],fi) and np.array_equal(indices['outer_valid'],vi);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);assert np.array_equal(indices['inner_train'],it) and np.array_equal(indices['inner_meta'],im) and not np.intersect1d(fi,vi).size;run=out/f'fold{k}';run.mkdir(exist_ok=True);rec={'status':'inner_neural_fit','started_utc':now(),'split':split,'fold':k,'plan_sha256':sha(ROOT/'research_w3_nested_neural_plan.json'),'runner_sha256':sha(__file__),'inner_train_rows':len(it),'inner_meta_rows':len(im),'outer_validation_rows':len(vi),'outer_validation_excluded_from_all_new_fits':True};dump(run/'result.json',rec)
  with threadpool_limits(6):
   nn,nr=neural(raw,y,df,it,im,run/'nn_inner');rec['nn_inner']=nr
   if a.split=='probe':rec.update(status='resource_probe_complete',score_evaluated=False,eligible_resources=True,completed_utc=now());dump(run/'result.json',rec);dump(ROOT/'research_w3_nested_neural_hardware.json',rec);return
   rec['status']='inner_cb10_fit';dump(run/'result.json',rec);cb,cr=deep_cb(raw,y,ext,teacher,it,im,cache,run);parts,audits=cached_inner(cache,raw,y,ext,teacher,im);p=np.column_stack(parts+[cb,nn]);o=outer_predictions(split,vi,df,fold);cp=ROOT/'research_artifacts'/('cb_deep_oof/raw_oof.npz' if split=='oof' else f'cb_deep_{split}/fold0/validation.npz');cz=np.load(cp)
   if split=='oof':assert np.array_equal(cz['id'],df.id) and np.array_equal(cz['target'],y) and np.array_equal(cz['fold'],fold);cd=cz['pred'][vi];nn_outer=old[vi];r=ref[vi]
   else:assert np.array_equal(cz['idx'],vi) and np.array_equal(cz['id'],df.id.iloc[vi]) and np.array_equal(cz['target'],y[vi]);cd=cz['pred'];nn_outer=old;r=ref
   outer=np.column_stack([o,cd,nn_outer]);rd=float(np.max(np.abs(outer@WEIGHTS-r)));assert rd<1e-12;rec.update(status='meta_fit',cached_inner_audit=audits,cb10_inner=cr,outer_v6_reconstruction_max_abs_diff=rd,methods={});dump(run/'result.json',rec)
   for method in methods:
    m=features(p,teacher[im],method=='residual_tree');q=features(outer,teacher[vi],method=='residual_tree')
    if method=='logistic':
     model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,solver='lbfgs'));model.fit(m,y[im]);pred=model.predict_proba(q)[:,1];joblib.dump(model,run/'meta_logistic.joblib');loaded=joblib.load(run/'meta_logistic.joblib');again=loaded.predict_proba(q)[:,1];details={'standardized_coefficients':model[-1].coef_.tolist(),'intercept':model[-1].intercept_.tolist()}
    else:
     config=frozen['residual_meta_params'];model=LGBMClassifier(**config);initial=logit(np.clip(p@WEIGHTS,1e-6,1-1e-6));outer_initial=logit(np.clip(r,1e-6,1-1e-6));model.fit(m,y[im],init_score=initial);pred=expit(model.booster_.predict(q,raw_score=True,num_threads=6)+outer_initial);model.booster_.save_model(str(run/'meta_residual_tree.txt'));loaded=Booster(model_file=str(run/'meta_residual_tree.txt'));again=expit(loaded.predict(q,raw_score=True,num_threads=6)+outer_initial);details={'params':config}
    diff=float(np.max(np.abs(pred-again)));assert diff<1e-12 and np.isfinite(pred).all();c=compare(y[vi],r,pred);allpred[method][vi]=pred;np.savez_compressed(run/f'{method}_validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,v6=r);rec['methods'][method]={'vs_v6':c,'model_reload_max_abs_diff':diff,**details};del model,loaded
  rec.update(status='complete',completed_utc=now());dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
  if split=='oof':
   c=rec['methods'][selected]['vs_v6'];comparisons.append(c)
   if c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comparisons,'completed_utc':now()});return
 vi=np.arange(len(y)) if split=='oof' else splits[0][1];res={'status':'complete','split':split,'methods':{},'completed_utc':now()}
 for method in methods:
  pred=allpred[method][vi];assert np.isfinite(pred).all();c=compare(y[vi],ref,pred);eligible=gate(c,split) and all(x['delta_auc']>0 for x in comparisons);res['methods'][method]={'vs_v6':c,'eligible':eligible};np.savez_compressed(out/f'{method}_validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,v6=ref,**({'fold':fold} if fold is not None else {}))
 if split=='primary':
  choices=[m for m in methods if res['methods'][m]['eligible']];chosen=max(choices,key=lambda m:res['methods'][m]['vs_v6']['candidate_auc']) if choices else None;selection={'eligible':bool(choices),'method':chosen,'selected_utc':now(),'plan_sha256':sha(ROOT/'research_w3_nested_neural_plan.json')};dump(ROOT/'research_w3_nested_neural_selection.json',selection);res['selected_method']=chosen;res['eligible_for_next']=bool(choices)
 else:res.update(selected_method=selected,eligible_for_next=res['methods'][selected]['eligible'],fold_comparisons=comparisons)
 dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_w3_nested_neural_last_failure.json',{'status':'failed','ended_utc':now(),'argv':os.sys.argv,'error':repr(e)});raise

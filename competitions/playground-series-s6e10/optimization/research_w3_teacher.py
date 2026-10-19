"""Source-only RealMLP teacher and frozen raw synthetic NN +teacher view."""
from research_w3_common import *
import time,argparse,gc,joblib,cloudpickle
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule
from research_gpu_value_features import ValueFeatures
from research_original import safe_original
from research_gpu_value import save_model,load_model
def fit_nn(X,y,cfg,out,rec):
 m=RealMLP_TD_Classifier(**cfg);old=TabNNModule.on_train_epoch_end;last=[time.monotonic()];rec['epochs']=[]
 def hook(module):
  old(module);torch.cuda.synchronize();t=time.monotonic();rec['epochs'].append({'epoch':int(module.current_epoch)+1,'seconds':t-last[0],'gpu_free_bytes':gpu_epoch_guard()});last[0]=t;dump(out/'result.json',rec);print(out.name,rec['epochs'][-1],flush=True)
 TabNNModule.on_train_epoch_end=hook;t=time.monotonic()
 try:
  with threadpool_limits(6):m.fit(X,y)
 finally:TabNNModule.on_train_epoch_end=old
 assert len(rec['epochs'])==cfg['n_epochs'] and not m.cv_alg_interface_.model.creator.is_cv and np.array_equal(m.classes_,[0,1]);rec.update(fit_seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());return m
def source():
 out=ROOT/'research_artifacts/w3_source_nn';out.mkdir(exist_ok=False);rec={'status':'running','started_utc':now(),'hardware':setup_gpu(),'config':plan()['source_nn_config']};dump(out/'result.json',rec)
 try:
  original,train,test,cols,audit=safe_original();rec['source_audit']=audit;y=original.satisfaction.astype('int8').to_numpy();tr=ValueFeatures('raw');X=tr.fit_transform(original[cols],y,pd.DataFrame(index=range(len(y))));joblib.dump(tr,out/'transformer.joblib');m=fit_nn(X,y,plan()['source_nn_config'],out,rec)
  for name,df in [('train',train),('test',test)]:
   p=m.predict_proba(tr.transform(df[cols],pd.DataFrame(index=range(len(df)))))[:,1];assert np.isfinite(p).all();np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),pred=p)
  m.to('cpu');save_model(m,out/'model.pkl');rec.update(status='complete',completed_utc=now(),model_sha256=sha(out/'model.pkl'),synthetic_targets_used_in_teacher=False,validation_scores_used=False,train_features_sha256=sha(out/'train_features.npz'),test_features_sha256=sha(out/'test_features.npz'));dump(out/'result.json',rec)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(out/'result.json',rec);raise
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['source','primary','confirm','oof'],required=True);a=ap.parse_args();start_guard()
 if a.split=='source':source();return
 hw=setup_gpu();df,raw,y,splits,ref,old,fold=load_split(a.split);z=np.load(ROOT/'research_artifacts/w3_source_nn/train_features.npz');assert np.array_equal(z['id'],df.id);q=np.clip(z['pred'],1e-6,1-1e-6);teacher=np.log(q/(1-q)).astype('float32');out=ROOT/'research_artifacts'/f'w3_teacher_nn_{a.split}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);comparisons=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'running','started_utc':now(),'hardware':hw,'config':plan()['synthetic_nn_config'],'source_features_sha256':sha(ROOT/'research_artifacts/w3_source_nn/train_features.npz')};dump(run/'result.json',rec)
  try:
   tr=ValueFeatures('raw');X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],pd.DataFrame(index=range(len(fi))));V=tr.transform(raw.iloc[vi].reset_index(drop=True),pd.DataFrame(index=range(len(vi))));X['source_nn_logit']=teacher[fi];V['source_nn_logit']=teacher[vi];joblib.dump(tr,run/'transformer.joblib');m=fit_nn(X,y[fi],plan()['synthetic_nn_config'],run,rec);pred=m.predict_proba(V)[:,1];m.to('cpu');save_model(m,run/'model.pkl');loaded=load_model(run/'model.pkl');loaded.to('cuda:0');again=loaded.predict_proba(V)[:,1];diff=float(np.max(np.abs(again-pred)));assert diff<1e-7
   r=ref[vi] if a.split=='oof' else ref;o=old[vi] if a.split=='oof' else old;candidate=.75*r+.25*pred;control=.75*r+.25*o;c=compare(y[vi],r,candidate);matched=compare(y[vi],control,candidate);comparisons.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,control=control);rec.update(status='complete',completed_utc=now(),auc=float(roc_auc_score(y[vi],pred)),vs_v6=c,vs_matched_control=matched,model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.pkl'));dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);del m,loaded,X,V;gc.collect();torch.cuda.empty_cache()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_full':False,'reason':'nonpositive fold','fold_comparisons':comparisons});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;control=.75*ref+.25*old;c=compare(y[vi],ref,candidate);matched=compare(y[vi],control,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,control=control,**({'fold':fold} if fold is not None else {}));eligible=gate(c,a.split) and matched['delta_auc']>0 and all(x['delta_auc']>0 for x in comparisons);res={'status':'complete','variant':'source_nn','split':a.split,'vs_v6':c,'vs_matched_control':matched,'fold_comparisons':comparisons,'eligible_for_next':eligible,'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

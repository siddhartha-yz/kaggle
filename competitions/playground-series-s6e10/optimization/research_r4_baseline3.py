"""Fresh complete baseline3 refit; old models never predict rows they trained on."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from research_r4_common import ROOT,now,dump,sha,guard
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,gc,joblib,cloudpickle,resource
import numpy as np,pandas as pd,lightgbm as lgb,torch
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier
from research_oof import params
from research_features import DistanceFeatures
from research_gpu_value_features import ValueFeatures
from research_external_ablation import external
OUT=ROOT/'research_artifacts/r4_baseline_replication3'
NAMES=['lgb','cb8','residual','cb10','nn']

def main():
 result=json.loads((ROOT/'research_artifacts/r4_lean_gpu_init_oof/result.json').read_text());assert result['status']=='complete' and result['eligible_for_new_partition_replication'],'No baseline refits without frozen fivefold gate'
 torch.set_num_threads(6);torch.set_num_interop_threads(1);OUT.mkdir(exist_ok=True);df=pd.read_csv(ROOT/'data/train.csv');assert sha(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36';y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();fold=np.full(len(y),-1,dtype='int8')
 for k,(_,vi) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261022).split(raw,y)):fold[vi]=k
 if (OUT/'folds.npz').exists():
  z=np.load(OUT/'folds.npz');assert all(np.array_equal(z[n],v) for n,v in [('id',df.id),('target',y),('fold',fold)])
 else:np.savez_compressed(OUT/'folds.npz',id=df.id.to_numpy(),target=y,fold=fold)
 cfg=dict(json.loads((ROOT/'research_gpu_value_plan.json').read_text())['config']);cfg.update(device='cpu',n_ens=2)
 frozen={'frozen_utc':now(),'folds_sha256':sha(OUT/'folds.npz'),'source_features_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/train_features.npz'),'features_sha256':sha(ROOT/'research_features.py'),'raw_nn_features_sha256':sha(ROOT/'research_gpu_value_features.py'),'nn_config':cfg,'weights':[45/128,9/128,18/128,24/128,32/128],'baseline_cpu_neural_note':'Same native two-member architecture/config as admitted baseline, deviceCPU only; no resource-specific init subsampling for this baseline. Floating point training differs bydevice.','labels_reused_warning':'Prospective fitting replication on reusedlabels, not untouchedtest'}
 if not (OUT/'plan.json').exists():dump(OUT/'plan.json',frozen)
 P=np.full((len(y),5),np.nan)
 for k in range(3):
  fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);assert not np.intersect1d(fi,vi).size;run=OUT/f'fold{k}';run.mkdir(exist_ok=True);tr=DistanceFeatures('distance_stats_numeric');X=tr.fit_transform_te(raw.iloc[fi].reset_index(drop=True),y[fi]);V=tr.transform_te(raw.iloc[vi].reset_index(drop=True))
  for c in ext:X[c]=ext[c].to_numpy()[fi];V[c]=ext[c].to_numpy()[vi]
  joblib.dump(tr,run/'transformer.joblib')
  for j,name in enumerate(NAMES):
   guard();path=run/name;path.mkdir(exist_ok=True);recfile=path/'result.json'
   if recfile.exists() and json.loads(recfile.read_text()).get('status')=='complete':
    rec=json.loads(recfile.read_text());assert rec['model_sha256']==sha(path/rec['model_file']);z=np.load(path/'validation.npz');assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);P[vi,j]=z['pred'];continue
   tick=time.monotonic();rec={'status':'training','started_utc':now(),'fold':k,'component':name,'train_rows':len(fi),'valid_rows':len(vi),'train_indices_sha256':__import__('hashlib').sha256(fi.tobytes()).hexdigest(),'no_outer_validation_in_fit':True};dump(recfile,rec);dump(ROOT/'research_r4_state.json',dict(rec,active_training=True,pid=os.getpid(),route='fresh_baseline_replication3'));print('START',k,name,flush=True)
   with threadpool_limits(6):
    if name in ['cb8','cb10']:
     config=params('cb1308') if name=='cb8' else dict(json.loads((ROOT/'research_cb_deep_plan.json').read_text())['cb_params'],iterations=813);CX=X.copy();CV=V.copy();cats=CX.select_dtypes('category').columns.tolist()
     for c in cats:CX[c]=CX[c].astype(str);CV[c]=CV[c].astype(str)
     model=CatBoostClassifier(**config);model.fit(CX,y[fi],cat_features=cats);pred=model.predict_proba(CV,thread_count=6)[:,1];mf='model.cbm';model.save_model(str(path/mf));loaded=CatBoostClassifier();loaded.load_model(str(path/mf));again=loaded.predict_proba(CV,thread_count=6)[:,1];del CX,CV
    elif name in ['lgb','residual']:
     config=params('stats639');kwargs={}
     if name=='residual':config['n_estimators']=581;kwargs['init_score']=.5*logit(np.clip(teacher[fi],1e-6,1-1e-6))
     model=lgb.LGBMClassifier(**config);model.fit(X,y[fi],**kwargs);mf='model.txt';model.booster_.save_model(str(path/mf));loaded=lgb.Booster(model_file=str(path/mf))
     if name=='residual':pred=expit(model.booster_.predict(V,raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[vi],1e-6,1-1e-6)));again=expit(loaded.predict(V,raw_score=True,num_threads=6)+.5*logit(np.clip(teacher[vi],1e-6,1-1e-6)))
     else:pred=model.predict_proba(V)[:,1];again=loaded.predict(V,num_threads=6)
    else:
     config=cfg;nt=ValueFeatures('raw');NX=nt.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],pd.DataFrame(index=range(len(fi))));NV=nt.transform(raw.iloc[vi].reset_index(drop=True),pd.DataFrame(index=range(len(vi))));model=RealMLP_TD_Classifier(**cfg);model.fit(NX,y[fi]);assert not model.cv_alg_interface_.model.creator.is_cv;pred=model.predict_proba(NV)[:,1];mf='model.pkl'
     with open(path/mf,'wb') as f:cloudpickle.dump(model,f)
     del model;gc.collect()
     with open(path/mf,'rb') as f:loaded=cloudpickle.load(f)
     again=loaded.predict_proba(NV.iloc[:2048])[:,1];model=loaded;joblib.dump(nt,path/'transformer.joblib');del NX,NV,nt
   diff=float(np.max(np.abs(pred[:len(again)]-again)));assert diff< (1e-7 if name=='nn' else 1e-12);assert np.isfinite(pred).all();P[vi,j]=pred;np.savez_compressed(path/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);rec.update(status='complete',ended_utc=now(),config=config,elapsed_seconds=time.monotonic()-tick,reload_max_diff=diff,model_file=mf,model_sha256=sha(path/mf),prediction_sha256=sha(path/'validation.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(recfile,rec);print('DONE',k,name,rec['elapsed_seconds'],flush=True);del model,loaded;gc.collect()
  del X,V,tr;gc.collect()
 assert np.isfinite(P).all();ref=P@np.array(frozen['weights']);np.savez_compressed(OUT/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,components=P,pred=ref,old_nn=P[:,4]);dump(OUT/'result.json',{'status':'complete','ended_utc':now(),'oof_sha256':sha(OUT/'oof.npz'),'fold_sha256':sha(OUT/'folds.npz'),'all_baseline_components_freshly_trained':True,'no_old_predictions_reused':True});print('BASELINE3_COMPLETE',flush=True)
if __name__=='__main__':main()

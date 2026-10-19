"""Cross-fitted domain-only density ratio and original-covariate PMI; no satisfaction labels."""
import numpy as np,pandas as pd,lightgbm as lgb,json,gc,hashlib
from pathlib import Path
from sklearn.model_selection import KFold
from research_features import DistanceFeatures
DOMAIN_PARAMS=dict(n_estimators=300,learning_rate=.05,num_leaves=31,max_depth=6,min_child_samples=200,reg_lambda=10,class_weight='balanced',objective='binary',n_jobs=6,random_state=20261024,verbosity=-1,deterministic=True,force_col_wise=True)
def logit(x):x=np.clip(x,1e-4,1-1e-4);return np.log(x/(1-x))
class DensityFeatures:
 def __init__(self,source,pairs,run):
  assert 'satisfaction' not in source and 'id' not in source;self.source=source;self.pairs=pairs;self.run=Path(run)
 def pair_fit(self):
  self.pmi=[];N=len(self.source)
  for a,b in self.pairs:
   u=self.source[a].to_numpy('int64');v=self.source[b].to_numpy('int64');assert min(u.min(),v.min())>=0 and max(u.max(),v.max())<=5
   joint=(np.bincount(u*6+v,minlength=36).reshape(6,6)+1)/(N+36);pa=(np.bincount(u,minlength=6)+1)/(N+6);pb=(np.bincount(v,minlength=6)+1)/(N+6);self.pmi.append(np.log(joint/(pa[:,None]*pb[None,:])))
 def pair_transform(self,raw):return np.mean([m[raw[a].to_numpy('int64'),raw[b].to_numpy('int64')] for (a,b),m in zip(self.pairs,self.pmi)],axis=0).astype('float32')
 def fit_domains(self,raw):
  assert 'satisfaction' not in raw and 'id' not in raw;assert list(raw)==list(self.source);self.pair_fit();self.domains=[];scores=np.full(len(raw),np.nan);coverage=np.zeros(len(raw),'int8');audit=[]
  for j,(it,iv) in enumerate(KFold(3,shuffle=True,random_state=20261024).split(raw)):
   assert not np.intersect1d(it,iv).size;coverage[iv]+=1;folder=self.run/f'domain{j}';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'indices.npz',train=it,valid=iv);combined=pd.concat([self.source,raw.iloc[it]],ignore_index=True);domain=np.r_[np.zeros(len(self.source),'int8'),np.ones(len(it),'int8')];tr=DistanceFeatures('base');X=tr.fit_transform_te(combined,domain);V=tr.transform_te(raw.iloc[iv].reset_index(drop=True));model=lgb.LGBMClassifier(**DOMAIN_PARAMS);model.fit(X,domain);p=model.predict_proba(V)[:,1];scores[iv]=logit(p);model.booster_.save_model(str(folder/'model.txt'));self.domains.append((tr,model.booster_));audit.append({'inner_fold':j,'source_rows':len(self.source),'synthetic_training_rows':len(it),'synthetic_heldout_rows':len(iv),'model_sha256':hashlib.sha256((folder/'model.txt').read_bytes()).hexdigest(),'domain_target_only':True,'satisfaction_used':False});del model,X,V,combined;gc.collect()
  assert np.all(coverage==1) and np.isfinite(scores).all();(self.run/'domain_audit.json').write_text(json.dumps(audit,indent=2)+'\n');self.source=None;return scores.astype('float32')
 def finish(self,raw,base,ext,scores):
  out=base.copy()
  for c in ext:out[c]=ext[c].to_numpy()
  out['domain_log_density_ratio']=scores;out['source_pair_mean_pmi']=self.pair_transform(raw);assert np.isfinite(out[['domain_log_density_ratio','source_pair_mean_pmi']]).all().all();return out
 def fit_transform(self,raw,y,ext):
  domain=self.fit_domains(raw);self.base=DistanceFeatures('distance_stats_numeric');return self.finish(raw,self.base.fit_transform_te(raw,y),ext,domain)
 def transform(self,raw,ext):
  scores=np.mean([logit(m.predict(tr.transform_te(raw),num_threads=6)) for tr,m in self.domains],axis=0).astype('float32');return self.finish(raw,self.base.transform_te(raw),ext,scores)

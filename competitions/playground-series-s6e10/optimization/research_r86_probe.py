"""Synthetic-only check: perfeature histogram resolution changes only teacher bins."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import numpy as np,pandas as pd,lightgbm as lgb,tempfile
from research_r83_common import ROOT,now,sha,dump
rng=np.random.default_rng(20261021);n=8192;x=rng.uniform(size=n);X=pd.DataFrame({'teacher':x,'other':rng.uniform(size=n),'nominal':pd.Categorical(rng.integers(0,3,n))});y=rng.binomial(1,.1+.8*x)
base=dict(max_bin=255,num_threads=2,verbosity=-1,data_random_seed=1)
a=lgb.Dataset(X,params=base,free_raw_data=False).construct();b=lgb.Dataset(X,params=dict(base,max_bin_by_feature=[1023,255,255]),free_raw_data=False).construct()
ba=[a.feature_num_bin(i) for i in range(3)];bb=[b.feature_num_bin(i) for i in range(3)];assert ba[0]==255 and bb[0]==1023 and ba[1:]==bb[1:]
m=lgb.LGBMClassifier(n_estimators=10,num_leaves=7,n_jobs=2,verbosity=-1,random_state=42,max_bin=255,max_bin_by_feature=[1023,255,255],deterministic=True,force_col_wise=True).fit(X,y);q=m.predict_proba(X.iloc[:128])[:,1]
assert m.booster_.params['max_bin_by_feature']==[1023,255,255]
with tempfile.TemporaryDirectory(prefix='r86_probe_',dir='/tmp') as d:
 path=d+'/model.txt';m.booster_.save_model(path);loaded=lgb.Booster(model_file=path);error=float(np.max(np.abs(loaded.predict(X.iloc[:128],num_threads=2)-q)));assert error<1e-12
out=dict(status='passed_synthetic_only',completed_utc=now(),source_sha256=sha(__file__),installed_LGBM_version=lgb.__version__,old_feature_bins=ba,new_feature_bins=bb,only_teacher_bins_changed=True,reload_error=error,official_data_or_labels_used=False,new_competition_fits=0)
dump(ROOT/'research_r86_probe.json',out);print(out)

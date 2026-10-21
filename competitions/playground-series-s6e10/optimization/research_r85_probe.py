"""Synthetic-only installed LightGBM monotone-constraint and reload check."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='2'
import numpy as np,pandas as pd,lightgbm as lgb,tempfile
from research_r83_common import ROOT,now,sha,dump
rng=np.random.default_rng(20261021);x=rng.uniform(size=512);noise=rng.normal(size=512);y=rng.binomial(1,.1+.8*x)
X=pd.DataFrame({'teacher':x,'other':noise});p=dict(n_estimators=20,num_leaves=7,min_child_samples=10,learning_rate=.1,verbosity=-1,n_jobs=2,random_state=42,deterministic=True,force_col_wise=True,monotone_constraints=[1,0]);m=lgb.LGBMClassifier(**p).fit(X,y)
grid=pd.DataFrame({'teacher':np.linspace(0,1,101),'other':np.zeros(101)});q=m.predict_proba(grid)[:,1];assert (np.diff(q)>=-1e-15).all() and q[-1]>q[0]
assert m.booster_.dump_model()['monotone_constraints']==[1,0]
with tempfile.TemporaryDirectory(prefix='r85_probe_',dir='/tmp') as d:
 path=d+'/model.txt';m.booster_.save_model(path);b=lgb.Booster(model_file=path);assert b.dump_model()['monotone_constraints']==[1,0];error=float(np.max(np.abs(b.predict(grid,num_threads=2)-q)));assert error<1e-12
out=dict(status='passed_synthetic_only',completed_utc=now(),source_sha256=sha(__file__),installed_LGBM_version=lgb.__version__,official_data_or_labels_used=False,new_competition_fits=0,monotone_vector_in_native_model=True,minimum_grid_delta=float(np.diff(q).min()),reload_error=error,method='InstalledLightGBMdefaultbasic; no methodorpenaltysearch')
dump(ROOT/'research_r85_probe.json',out);print(out)

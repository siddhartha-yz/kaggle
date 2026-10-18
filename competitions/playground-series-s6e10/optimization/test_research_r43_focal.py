"""Checkanalyticderivatives, gamma0BCEreduction andpointwiseBayescalibrationidentity."""
import numpy as np
from scipy.special import expit
from research_r43_focal import loss,gradients,posterior,focal_objective
z=np.array([-7.,-2.,-.2,.2,2.,7.]*2);y=np.repeat([0.,1.],6);eps=1e-5
g,h=gradients(y,z);fd=(loss(y,z+eps)-loss(y,z-eps))/(2*eps);hh=(gradients(y,z+eps)[0]-gradients(y,z-eps)[0])/(2*eps)
assert np.max(np.abs(g-fd))<1e-8 and np.max(np.abs(h-hh))<1e-8
b,bb=gradients(y,z,gamma=0);assert np.max(np.abs(b-(expit(z)-y)))<1e-14 and np.max(np.abs(bb-expit(z)*expit(-z)))<1e-14
assert np.max(np.abs(posterior(z,gamma=0)-expit(z)))<1e-14
grid=np.linspace(-8,8,1001);r=posterior(grid);gp=gradients(np.ones(len(grid)),grid)[0];gn=gradients(np.zeros(len(grid)),grid)[0]
assert np.max(np.abs(r*gp+(1-r)*gn))<1e-14 and (np.diff(r)>0).all() and np.max(np.abs(r+posterior(-grid)-1))<1e-14
extreme=np.array([-1000.,-50.,0.,50.,1000.]);a,c=focal_objective([1,1,0,0,0],extreme);assert np.isfinite(a).all() and (c>=1e-3).all() and np.isfinite(posterior(extreme)).all()
print('PASS analyticgradient/Hessian, BCEgamma0, Bayesposterioridentity/monotonicity/symmetry, extreme-marginsandpositive-Hessianfloor')
# Tiny synthetic CPU-only API/reload contract, not a competition quality experiment.
import lightgbm as lgb
from pathlib import Path
rng=np.random.default_rng(20261043);X=rng.normal(size=(256,3));Y=(X[:,0]+.2*X[:,1]>0).astype('int8');history={}
m=lgb.LGBMClassifier(objective=focal_objective,n_estimators=8,num_leaves=7,min_child_samples=5,learning_rate=.05,n_jobs=1,verbosity=-1,boost_from_average=False,max_delta_step=1,metric='auc')
m.fit(X[:128],Y[:128],eval_set=[(X[128:],Y[128:])],eval_metric='auc',callbacks=[lgb.record_evaluation(history)]);assert np.array_equal(m.classes_,[0,1]) and list(history['valid_0'])==['auc']
raw=m.booster_.predict(X,raw_score=True,num_threads=1);p=posterior(raw);path=Path('/tmp/kaggle_r43_synthetic_contract.txt');m.booster_.save_model(str(path));re=lgb.Booster(model_file=str(path));assert np.max(np.abs(p-posterior(re.predict(X,raw_score=True,num_threads=1))))<1e-15
print('PASS tiny synthetic LightGBM customobjective/rawmargin/metric/classorder/reloadcontract; no officialdataread')

"""Synthetic calculus, convexity, global optimum and exact v12 identity checks."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
import json,numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from research_r69_gate import PairGate,design,objective
from research_r67_pairs import pairs
rng=np.random.default_rng(20261069);n=120;y=np.r_[np.zeros(60,dtype='int8'),np.ones(60,dtype='int8')];t=expit(rng.normal(size=n));p=expit(rng.normal(size=n));r=expit(rng.normal(size=n));m,x=design(t,p,r);g,w=pairs(y,32,20261069)
a=objective(.13,m,x,g,w,32,100);h=1e-5;lo=objective(.13-h,m,x,g,w,32,100);hi=objective(.13+h,m,x,g,w,32,100)
assert abs((hi[0]-lo[0])/(2*h)-a[1])<1e-6
assert abs((hi[1]-lo[1])/(2*h)-a[2])<1e-5
fit=PairGate().fit(t,p,r,y);ind=minimize_scalar(lambda z:objective(z,m,x,g,w,32,100)[0],bounds=(-.5,.5),method='bounded',options={'xatol':1e-12})
assert abs(fit.theta-ind.x)<1e-6
zero=PairGate();zero.theta=0;baseline=expit(.875*logit(np.clip(expit(.5*(logit(t)+logit(p))),1e-6,1-1e-6))+.125*logit(r));err=float(np.max(np.abs(zero.predict(t,p,r)-baseline)));assert err<1e-12
assert all(objective(v,m,x,g,w,32,100)[2]>=100 for v in [-.5,0,.5])
print(json.dumps({'status':'passed_synthetic_only','finite_difference_gradient_error':abs((hi[0]-lo[0])/(2*h)-a[1]),'finite_difference_hessian_error':abs((hi[1]-lo[1])/(2*h)-a[2]),'independent_optimum_error':abs(fit.theta-ind.x),'v12_zero_gate_identity_error':err,'KKT_residual':fit.optimization['KKT_residual']}))

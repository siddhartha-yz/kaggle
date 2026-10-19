"""Per-member additive-origin invariance and training-prior identification, synthetic only."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json,numpy as np
from research_r70_member_origin import fit_origins,predict_calibrated
rng=np.random.default_rng(20261070);y=np.r_[np.zeros(400,dtype='int8'),np.ones(600,dtype='int8')];m=rng.normal(size=(32,1000));v=rng.normal(size=(32,200));shift=rng.normal(0,10,size=32);a=fit_origins(m,y);b=fit_origins(m+shift[:,None],y);err=float(np.max(np.abs(predict_calibrated(v,a)-predict_calibrated(v+shift[:,None],b))));origin=float(np.max(np.abs(a-(b+shift))));prior=float(abs(predict_calibrated(m,a).mean()-y.mean()));assert err<1e-12 and origin<1e-10 and prior<1e-12
print(json.dumps({'status':'passed_synthetic_only','member_origin_shift_invariance_error':err,'intercept_equivariance_error':origin,'training_prior_error':prior}))

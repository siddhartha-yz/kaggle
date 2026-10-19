"""Independent gradient and convex-opinion-pool bound checks on synthetic data."""
import numpy as np,json
from scipy.special import expit,logit
from research_r65_gate import design,objective,ConfidenceGate
rng=np.random.default_rng(20261065);tree=expit(rng.normal(0,3,1000));nn=expit(rng.normal(0,3,1000));y=rng.integers(0,2,1000);beta=np.array([.03,-.04,.02]);m,X=design(tree,nn);loss,g=objective(beta,m,X,y,100.)
eps=1e-6;numeric=np.array([(objective(beta+np.eye(3)[j]*eps,m,X,y,100.)[0]-objective(beta-np.eye(3)[j]*eps,m,X,y,100.)[0])/(2*eps) for j in range(3)]);assert np.max(np.abs(g-numeric))<1e-6
for b in [np.array([i,j,k])/6 for i in [-1,1] for j in [-1,1] for k in [-1,1]]:
 p=expit(m+X@b);assert (p>=np.minimum(tree,nn)-1e-14).all() and (p<=np.maximum(tree,nn)+1e-14).all()
fit=ConfidenceGate().fit(tree,nn,y);assert fit.optimization['objective']<=fit.optimization['initial_objective'];assert np.isfinite(fit.predict(tree,nn)).all()
print(json.dumps(dict(status='passed_synthetic_no_official_data',gradient_max_error=float(np.max(np.abs(g-numeric))),box_corners_checked=8,rows=1000)))

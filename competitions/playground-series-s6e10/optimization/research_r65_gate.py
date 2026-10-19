"""A three-coefficient bounded confidence gate, fitted only on honest inner expert predictions."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize

def design(tree,nn):
    t=logit(np.clip(tree,1e-6,1-1e-6));n=logit(np.clip(nn,1e-6,1-1e-6));m=.5*(t+n);u=np.tanh(m/2)
    return m,(n-t)[:,None]*np.column_stack([np.ones(len(m)),u,.5*(3*u*u-1)])
def objective(beta,offset,X,y,ridge):
    eta=offset+X@beta
    return float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*ridge*(beta@beta)),X.T@(expit(eta)-y)+ridge*beta
class ConfidenceGate:
    def __init__(self,ridge=100.):self.ridge=ridge
    def fit(self,tree,nn,y):
        m,X=design(tree,nn);zero=np.zeros(3)
        r=minimize(objective,zero,args=(m,X,np.asarray(y),self.ridge),jac=True,method='L-BFGS-B',bounds=[(-1/6,1/6)]*3,options=dict(maxiter=1000,ftol=1e-12,gtol=1e-6,maxls=50))
        assert r.success and np.isfinite(r.x).all() and np.max(np.abs(r.x))<=1/6,str(r.message)
        self.beta=r.x;self.optimization=dict(success=bool(r.success),message=str(r.message),iterations=int(r.nit),objective=float(r.fun),initial_objective=objective(zero,m,X,np.asarray(y),self.ridge)[0]);assert self.optimization['objective']<=self.optimization['initial_objective']+1e-8
        return self
    def predict(self,tree,nn):
        m,X=design(tree,nn);return expit(m+X@self.beta)

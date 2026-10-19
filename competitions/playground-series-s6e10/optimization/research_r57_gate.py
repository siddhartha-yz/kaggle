"""Bounded cohort log-odds mixture, ridge centered on the fixed v11 equal pool."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
COLS=['Class','Type of Travel','Customer Type']
def keys(raw):return raw[COLS].astype(str).agg('\x1f'.join,axis=1).to_numpy()
def logits(tree,nn):
    t=logit(np.clip(tree,1e-6,1-1e-6));n=logit(np.clip(nn,1e-6,1-1e-6));return .5*(t+n),n-t
def objective(theta,base,delta,index,y,ridge):
    eta=base+theta[index]*delta;p=expit(eta)
    loss=float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*ridge*(theta@theta))
    gradient=np.bincount(index,weights=(p-y)*delta,minlength=len(theta))+ridge*theta
    return loss,gradient
class CohortGate:
    def __init__(self,ridge=100.):self.ridge=ridge
    def fit(self,raw,tree,nn,y):
        k=keys(raw);self.vocab,index=np.unique(k,return_inverse=True);base,delta=logits(tree,nn);zero=np.zeros(len(self.vocab))
        result=minimize(objective,zero,args=(base,delta,index,np.asarray(y),self.ridge),method='L-BFGS-B',jac=True,bounds=[(-.5,.5)]*len(zero),options=dict(maxiter=1000,ftol=1e-12,gtol=1e-6,maxls=50))
        assert result.success and np.isfinite(result.x).all() and (np.abs(result.x)<=.5).all(),str(result.message)
        self.theta=result.x;self.optimization=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),objective=float(result.fun),initial_objective=objective(zero,base,delta,index,np.asarray(y),self.ridge)[0]);assert self.optimization['objective']<=self.optimization['initial_objective']+1e-8
        return self
    def coefficients(self,raw):
        mapping=dict(zip(self.vocab,self.theta));return np.array([mapping.get(k,0.) for k in keys(raw)])
    def predict(self,raw,tree,nn):
        base,delta=logits(tree,nn);return expit(base+self.coefficients(raw)*delta)

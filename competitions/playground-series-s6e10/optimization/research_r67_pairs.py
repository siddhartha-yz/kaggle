"""A bounded unbiased bipartite pair graph and intercept identification for PairLogit."""
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq

def pairs(y,k=8,seed=20261067):
    y=np.asarray(y);assert y.ndim==1 and np.array_equal(np.unique(y),[0,1]);positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0);rng=np.random.default_rng(seed)
    winner=np.r_[np.repeat(positive,k),positive[rng.integers(len(positive),size=len(negative)*k)]]
    loser=np.r_[negative[rng.integers(len(negative),size=len(positive)*k)],np.repeat(negative,k)]
    encoded=winner.astype('int64')*len(y)+loser;unique,counts=np.unique(encoded,return_counts=True);graph=np.column_stack([unique//len(y),unique%len(y)]).astype('int32');weights=counts.astype('float64')
    assert weights.sum()==len(y)*k and (y[graph[:,0]]==1).all() and (y[graph[:,1]]==0).all()
    degree=np.bincount(graph.ravel(),weights=np.repeat(weights,2),minlength=len(y));assert degree.min()>=k
    return graph,weights

def intercept(raw,y):
    raw=np.asarray(raw,'float64');prior=float(np.mean(y));assert np.isfinite(raw).all() and 0<prior<1
    def error(b):return float(expit(raw+b).mean()-prior)
    bias=float(brentq(error,-float(np.max(raw))-50,-float(np.min(raw))+50,xtol=1e-12));assert abs(error(bias))<1e-12
    return bias

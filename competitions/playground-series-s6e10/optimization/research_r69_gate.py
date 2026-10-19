"""One scalar convex pairwise-logistic expert gate, preserving the fixed RankNet share."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq
from research_r67_pairs import pairs

def design(tree,nn,ranknet):
    t,n,r=[logit(np.clip(np.asarray(p,'float64'),1e-6,1-1e-6)) for p in (tree,nn,ranknet)]
    return .875*.5*(t+n)+.125*r,.875*(n-t)

def objective(theta,offset,direction,graph,weights,k,ridge):
    i,j=graph.T;z=(offset[i]-offset[j])+theta*(direction[i]-direction[j]);x=direction[i]-direction[j];q=expit(-z)
    return (float(weights@np.logaddexp(0,-z)/k+.5*ridge*theta*theta),float(-(weights*q)@x/k+ridge*theta),float((weights*q*(1-q))@(x*x)/k+ridge))

class PairGate:
    def __init__(self,ridge=100.,k=32,seed=20261069):self.ridge=ridge;self.k=k;self.seed=seed
    def fit(self,tree,nn,ranknet,y):
        y=np.asarray(y);assert np.array_equal(np.unique(y),[0,1]);m,x=design(tree,nn,ranknet);g,w=pairs(y,self.k,self.seed)
        def f(t):return objective(t,m,x,g,w,self.k,self.ridge)
        left,right=f(-.5)[1],f(.5)[1]
        theta=-.5 if left>=0 else .5 if right<=0 else float(brentq(lambda t:f(t)[1],-.5,.5,xtol=1e-13))
        loss,grad,hessian=f(theta);initial=f(0)[0]
        assert hessian>=self.ridge and loss<=initial+1e-8
        kkt=abs(grad) if -.5<theta<.5 else max(0.,-grad) if theta==-.5 else max(0.,grad)
        assert kkt<1e-7
        self.theta=theta;self.optimization=dict(objective=loss,initial_objective=initial,gradient=grad,hessian=hessian,KKT_residual=kkt,weighted_pairs=float(w.sum()),unique_pairs=len(g),rows=len(y),success=True)
        return self
    def predict(self,tree,nn,ranknet):
        m,x=design(tree,nn,ranknet);return expit(m+self.theta*x)

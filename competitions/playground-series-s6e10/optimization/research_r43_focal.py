"""Binarygamma1focalloss andanalyticBayesposteriorcorrection, stablefinite margins."""
import numpy as np
from scipy.special import expit
GAMMA=1.0
HESSIAN_FLOOR=1e-3

def loss(labels,margin,gamma=GAMMA):
    y=np.asarray(labels,dtype='float64');z=np.asarray(margin,dtype='float64');t=2*y-1
    return expit(-t*z)**gamma*np.logaddexp(0,-t*z)

def gradients(labels,margin,gamma=GAMMA,hessian_floor=None):
    y=np.asarray(labels,dtype='float64');z=np.asarray(margin,dtype='float64');assert y.shape==z.shape and z.ndim==1 and np.isin(y,[0,1]).all() and np.isfinite(z).all()
    t=2*y-1;p=expit(t*z);q=expit(-t*z);ce=np.logaddexp(0,-t*z);a=q**gamma
    grad=-t*a*(q+gamma*p*ce)
    hess=p*a*((2*gamma+1)*q+gamma*(gamma*p-q)*ce)
    if hessian_floor is not None:hess=np.maximum(hess,hessian_floor)
    assert np.isfinite(grad).all() and np.isfinite(hess).all();return grad,hess

def focal_objective(labels,margin):return gradients(labels,margin,hessian_floor=HESSIAN_FLOOR)

def posterior(margin,gamma=GAMMA):
    z=np.asarray(margin,dtype='float64');assert np.isfinite(z).all();p=expit(z);q=expit(-z)
    positive_magnitude=q**gamma*(q+gamma*p*np.logaddexp(0,-z))
    negative_gradient=p**gamma*(p+gamma*q*np.logaddexp(0,z))
    probability=negative_gradient/(positive_magnitude+negative_gradient)
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all();return probability

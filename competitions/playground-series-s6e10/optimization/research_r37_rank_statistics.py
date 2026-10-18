"""Signed conditional mid-ranks for ordinal positive ratings; state0 treated separately from the ordered positive states."""
import numpy as np
from scipy.special import ndtri

def conditional_normal_rank(probabilities,classes,observed):
    p=np.asarray(probabilities,dtype='float64');classes=np.asarray(classes);r=np.asarray(observed)
    assert p.ndim==2 and p.shape==(len(r),len(classes)) and np.isfinite(p).all() and (p>=0).all()
    assert len(np.unique(classes))==len(classes) and np.isin(classes,np.arange(6)).all() and np.isin(r,np.arange(6)).all()
    assert np.max(np.abs(p.sum(1)-1))<2e-6
    positive=classes>0;mass=p[:,positive].sum(1)
    numerator=(p*((classes[None,:]>0)&(classes[None,:]<r[:,None]))).sum(1)+.5*(p*(classes[None,:]==r[:,None])).sum(1)
    u=np.full(len(r),.5);np.divide(numerator,mass,out=u,where=(r>0)&(mass>1e-12));u[(r==0)|(mass<=1e-12)]=.5
    result=ndtri(np.clip(u,1e-6,1-1e-6)).astype('float32');assert np.isfinite(result).all();return result

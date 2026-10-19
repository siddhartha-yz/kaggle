"""Information retained by conditional class probabilities beyond a rating mean."""
import numpy as np
EPS=1e-6
def conditional_statistics(probability,classes,observed):
    p=np.asarray(probability,dtype='float64');c=np.asarray(classes,dtype='float64');x=np.asarray(observed,dtype='float64')
    assert p.ndim==2 and c.ndim==1 and x.ndim==1 and p.shape==(len(x),len(c)) and len(np.unique(c))==len(c)
    assert np.isfinite(p).all() and np.isfinite(c).all() and np.isfinite(x).all() and ((p>=0)&(p<=1)).all()
    assert np.max(np.abs(p.sum(1)-1))<1e-6
    expected=(p@c).astype('float32')
    observed_probability=np.sum(p*(x[:,None]==c[None,:]),axis=1)
    surprisal=(-np.log(np.clip(observed_probability,EPS,1))).astype('float32')
    return expected,surprisal

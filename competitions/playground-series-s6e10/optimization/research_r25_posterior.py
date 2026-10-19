"""Class-identity-preserving rating posteriors, fixed six known rating states."""
import numpy as np

def posterior_features(probability,classes):
    p=np.asarray(probability,dtype='float64');c=np.asarray(classes,dtype='float64')
    assert p.ndim==2 and c.ndim==1 and p.shape[1]==len(c) and len(np.unique(c))==len(c)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(1)-1))<1e-6
    assert np.isfinite(c).all() and np.array_equal(c,c.astype('int64')) and np.isin(c,np.arange(6)).all()
    out=np.zeros((len(p),6),'float32');out[:,c.astype('int64')]=p
    assert np.max(np.abs(out.sum(1,dtype='float64')-1))<1e-6
    return out

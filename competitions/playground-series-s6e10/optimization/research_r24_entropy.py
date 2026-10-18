"""Conditional uncertainty and centered self-surprisal; no target labels."""
import numpy as np
from research_r21_conditional import conditional_statistics

def uncertainty(probability,classes,observed):
    expected,surprisal=conditional_statistics(probability,classes,observed)
    p=np.asarray(probability,dtype='float64')
    entropy=-np.sum(np.where(p>0,p*np.log(np.clip(p,1e-300,1)),0),axis=1)
    assert np.isfinite(entropy).all() and (entropy>=-1e-12).all() and (entropy<=np.log(p.shape[1])+1e-6).all()
    return expected,surprisal,entropy.astype('float32'),(surprisal.astype('float64')-entropy).astype('float32')

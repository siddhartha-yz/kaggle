"""Inference-only native member logits and independently identified training score origins."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq

def _identity(self,output):return output

@contextmanager
def native_logits(model):
    module=model.cv_alg_interface_.model
    assert module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3
    assert '_postprocess_ens_pred' not in vars(module)
    module._postprocess_ens_pred=MethodType(_identity,module)
    try:yield
    finally:del module._postprocess_ens_pred

def predict_margins(model,frame):
    result=np.empty((32,len(frame)),'float64')
    with native_logits(model):
        for lo in range(0,len(frame),2048):
            hi=min(lo+2048,len(frame));logits=model._predict_raw(frame.iloc[lo:hi]);assert list(logits.shape)==[32,hi-lo,2]
            raw=logits.numpy().astype('float64');result[:,lo:hi]=raw[:,:,1]-raw[:,:,0]
    assert np.isfinite(result).all();return result

def fit_origins(margin,y):
    margin=np.asarray(margin,'float64');y=np.asarray(y);assert margin.shape==(32,len(y)) and np.isfinite(margin).all() and np.array_equal(np.unique(y),[0,1]);prior=float(y.mean());out=[]
    for row in margin:
        def error(b):return float(expit(row+b).mean()-prior)
        b=float(brentq(error,-float(row.max())-50,-float(row.min())+50,xtol=1e-12));assert abs(error(b))<1e-12;out.append(b)
    return np.asarray(out,'float64')

def predict_calibrated(margin,bias):
    assert np.asarray(margin).shape[0]==len(bias)==32
    return expit(np.asarray(margin)+np.asarray(bias)[:,None]).mean(axis=0)

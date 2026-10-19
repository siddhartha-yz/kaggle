"""Bounded query batches and training-only origin identification for a ranking score."""
import numpy as np
from scipy.special import expit,logit
from research_r67_pairs import intercept
EPS=1e-6
CHUNK=2048

def raw_score(probability):return logit(np.clip(np.asarray(probability,'float64'),EPS,1-EPS))
def calibrate(probability,bias):return expit(raw_score(probability)+bias)
def fit_bias(probability,y):return intercept(raw_score(probability),y)
def predict_raw(model,frame):
    assert np.array_equal(model.classes_,[0,1]);pred=np.empty(len(frame),'float64')
    for lo in range(0,len(frame),CHUNK):
        hi=min(lo+CHUNK,len(frame));pred[lo:hi]=model.predict_proba(frame.iloc[lo:hi])[:,1]
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();return pred

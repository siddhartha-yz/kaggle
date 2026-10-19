"""Prospective CPU-only EBM family; no execution while another model job runs."""
from pathlib import Path
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parent
import numpy as np,pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

def parameters(rounds=50000):
 return dict(max_bins=1024,max_interaction_bins=64,interactions='3x',validation_size=.15,outer_bags=8,inner_bags=0,learning_rate=.015,greedy_ratio=10.,cyclic_progress=False,smoothing_rounds=75,interaction_smoothing_rounds=75,max_rounds=rounds,early_stopping_rounds=100,early_stopping_tolerance=1e-5,min_samples_leaf=4,min_hessian=.0001,reg_alpha=0.,reg_lambda=0.,max_delta_step=0.,gain_scale=5.,min_cat_samples=10,cat_smooth=10.,missing='separate',max_leaves=2,objective='log_loss',n_jobs=8,random_state=42)

def make_model(raw,rounds=50000):
 assert raw.shape[1]==21 and 'id' not in raw and 'satisfaction' not in raw
 nominal=list(raw.select_dtypes(exclude='number'));assert len(nominal)==4
 return ExplainableBoostingClassifier(feature_names=list(raw),feature_types=['nominal' if c in nominal else 'continuous' for c in raw],**parameters(rounds))

def reconstruct_probability(model,raw):
 scores=model.eval_terms(raw).sum(axis=1)+float(np.asarray(model.intercept_).reshape(-1)[0])
 return 1./(1.+np.exp(-scores))

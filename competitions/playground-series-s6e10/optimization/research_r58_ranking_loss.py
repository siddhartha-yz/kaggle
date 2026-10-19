"""Standby: one fixed .9 supervised CE +.1 minibatch positive-negative RankNet loss."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from pytabkit.models.training.metrics import cross_entropy
from pytabkit.models.training.nn_creator import NNCreator
COUNTERS={'calls':0,'first_logits_shape':None,'first_target_shape':None}

def ranking_criterion(y_pred,y):
    assert y_pred.ndim==3 and y_pred.shape[-1]==2 and y.shape[:-1]==y_pred.shape[:-1]
    ce=cross_entropy(y_pred,y);hard=y.argmax(dim=-1) if torch.is_floating_point(y) else y.squeeze(-1)
    score=y_pred[...,1]-y_pred[...,0];difference=score.unsqueeze(-1)-score.unsqueeze(-2);mask=(hard.unsqueeze(-1)==1)&(hard.unsqueeze(-2)==0)
    count=mask.sum(dim=(-2,-1)).clamp_min(1);pairs=torch.where(mask,F.softplus(-difference),0.).sum(dim=(-2,-1))/count
    assert pairs.shape==ce.shape
    COUNTERS['calls']+=1
    if COUNTERS['first_logits_shape'] is None:COUNTERS.update(first_logits_shape=list(y_pred.shape),first_target_shape=list(y.shape))
    return .9*ce+.1*pairs

@contextmanager
def ranking_backend():
    original=NNCreator.get_criterions
    def get(self):
        criterion,validation=original(self);assert self.n_classes==2 and criterion is cross_entropy
        return ranking_criterion,validation
    NNCreator.get_criterions=get
    try:yield
    finally:NNCreator.get_criterions=original

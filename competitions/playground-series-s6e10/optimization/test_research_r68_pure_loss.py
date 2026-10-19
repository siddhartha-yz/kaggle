"""Independent explicit-pair gradients, score-origin invariance and degenerate batches."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,numpy as np,torch
from torch.nn import functional as F
from scipy.special import expit
from research_r68_pure_loss import pure_ranknet
from research_r68_calibration import fit_bias,calibrate

torch.set_num_threads(6);torch.manual_seed(20261068);logits=torch.randn(3,7,2,dtype=torch.float64,requires_grad=True);hard=torch.tensor([[0,1,1,0,1,0,1],[1,0,0,1,0,1,0],[1,1,0,0,1,1,0]]);labels=hard.unsqueeze(-1);actual=pure_ranknet(logits,labels);independent=[]
for m in range(3):
    terms=[]
    for i in range(7):
        for j in range(7):
            if hard[m,i]==1 and hard[m,j]==0:terms.append(F.softplus(-(logits[m,i,1]-logits[m,i,0]-logits[m,j,1]+logits[m,j,0])))
    independent.append(torch.stack(terms).mean())
expected=torch.stack(independent);g=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0];eg=torch.autograd.grad(expected.sum(),logits)[0];assert torch.max(torch.abs(actual-expected))<1e-12 and torch.max(torch.abs(g-eg))<1e-12 and torch.max(torch.abs(g[:,:,1].sum(1)))<1e-12
shifted=logits.detach().clone();shifted[:,:,1]+=3;assert torch.max(torch.abs(pure_ranknet(shifted,labels)-actual))<1e-12
smooth=F.one_hot(hard,2).double()*.99+.005;assert torch.equal(pure_ranknet(logits.detach(),smooth),actual.detach())
for c in [0,1]:
    x=logits.detach().clone().requires_grad_(True);loss=pure_ranknet(x,torch.full((3,7,1),c));assert torch.equal(loss,torch.zeros_like(loss));assert torch.equal(torch.autograd.grad(loss.sum(),x)[0],torch.zeros_like(x))
rng=np.random.default_rng(20261068);score=rng.normal(size=1000);target=rng.integers(0,2,1000);b=fit_bias(expit(score),target);other=fit_bias(expit(score+2),target);error=float(np.max(np.abs(calibrate(expit(score),b)-calibrate(expit(score+2),other))));assert error<1e-12 and abs(calibrate(expit(score),b).mean()-target.mean())<1e-12
print(json.dumps(dict(status='passed_no_official_data',loss_error=float(torch.max(torch.abs(actual-expected))),gradient_error=float(torch.max(torch.abs(g-eg))),class1_bias_gradient_sum_max=float(torch.max(torch.abs(g[:,:,1].sum(1)))),calibrated_origin_invariance_error=error,CE_coefficient=0.,pairwise_coefficient=1.,oneclass_loss_and_gradient_zero=True,smoothed_and_integer_hardlabels_equal=True)))

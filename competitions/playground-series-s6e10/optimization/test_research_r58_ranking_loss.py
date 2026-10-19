"""No official data: exact loss/gradient vs explicit pairs, invariances and one-class safety."""
import torch
from torch.nn import functional as F
from pytabkit.models.training.metrics import cross_entropy
from research_r58_ranking_loss import ranking_criterion
def main():
    torch.set_num_threads(6);torch.manual_seed(20261058);x=torch.randn(2,6,2,dtype=torch.float64,requires_grad=True);hard=torch.tensor([[0,1,0,1,0,1],[1,1,0,0,1,0]]);y=.99*F.one_hot(hard,2).double()+.005
    actual=ranking_criterion(x,y);scores=x[...,1]-x[...,0];parts=[]
    for k in range(2):
        pairs=[F.softplus(-(scores[k,i]-scores[k,j])) for i in range(6) for j in range(6) if hard[k,i]==1 and hard[k,j]==0];parts.append(torch.stack(pairs).mean())
    expected=.9*cross_entropy(x,y)+.1*torch.stack(parts);loss_error=float((actual-expected).abs().max().detach());ga=torch.autograd.grad(actual.sum(),x,retain_graph=True)[0];gb=torch.autograd.grad(expected.sum(),x)[0];gradient_error=float((ga-gb).abs().max());assert loss_error<1e-12 and gradient_error<1e-12
    detached=x.detach();perm=torch.tensor([5,0,4,1,3,2]);assert torch.max(torch.abs(ranking_criterion(detached[:,perm],y[:,perm])-actual.detach()))<1e-12;assert torch.max(torch.abs(ranking_criterion(detached+100,y)-actual.detach()))<1e-12
    changed=detached.clone();changed[1]*=5;assert torch.equal(ranking_criterion(changed,y)[:1],actual.detach()[:1])
    only=F.one_hot(torch.ones((2,6),dtype=torch.long),2).double();assert torch.equal(ranking_criterion(detached,only),.9*cross_entropy(detached,only));assert torch.max(torch.abs(ranking_criterion(detached,hard.unsqueeze(-1))-(.9*cross_entropy(detached,hard.unsqueeze(-1))+.1*torch.stack(parts).detach())))<1e-12
    print('R58_SYNTHETIC_LOSS_TEST_PASSED',loss_error,gradient_error,flush=True)
if __name__=='__main__':main()

"""Check classdirection and analytic initial noise-link probabilities before training."""
import numpy as np,torch
from research_w3_noise_factor import DeepFM
def main():
 torch.set_num_threads(1);arch={'embedding_dim':16,'embedding_init_std':.01,'hidden':[512,256,128],'dropout':.1,'max_noise_rate':.25,'base_rates':[.1,.2]};m=DeepFM([2,3],2,arch);m.eval();n=torch.zeros(3,2);c=torch.zeros(3,2,dtype=torch.long);prior=torch.tensor([-20.,0.,20.]);p=m(n,c,prior).sigmoid().detach().numpy();expected=.1+.7*prior.sigmoid().numpy();assert np.allclose(p,expected,atol=2e-7) and np.allclose(p,[.1,.45,.8],atol=2e-7);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(n,c,prior),torch.tensor([1.,1.,0.]));loss.backward();assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None);print('PASS: q0=>false-to-true rate.1,q1=>1-true-to-false rate.8,q.5=>.45; finite BCE gradients')
if __name__=='__main__':main()

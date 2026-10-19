import inspect,itertools,unittest
import torch
from research_r73_density_model import JointDensity
class DensityTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(123);self.m=JointDensity([2,3],number_ratings=3,number_values=2,width=16,layers=2).eval();self.c=torch.tensor([[1,2]]*8);self.n=torch.randn(1,8).repeat(8,1);self.r=torch.tensor(list(itertools.product(range(2),repeat=3)))
 def test_joint_normalizes_and_Bayes_is_exact(self):
  with torch.no_grad():
   j=self.m.log_joints(self.c,self.n,self.r);p=self.m.predict_proba(self.c,self.n,self.r)
  self.assertLess(abs(j.exp().sum().item()-1),1e-6);self.assertLess(torch.max(torch.abs(p-j.softmax(1)[:,1])).item(),1e-6)
 def test_current_and_future_ratings_are_hidden(self):
  for position in range(3):
   changed=self.r.clone();changed[:,position:]=1-changed[:,position:]
   with torch.no_grad():
    a=self.m.density_logits(self.c,self.n,self.r,torch.ones(8,dtype=torch.long));b=self.m.density_logits(self.c,self.n,changed,torch.ones(8,dtype=torch.long))
   self.assertTrue(torch.equal(a[0],b[0]));self.assertLess((a[1][:,:position+1]-b[1][:,:position+1]).abs().max().item(),1e-6)
 def test_prior_does_not_see_condition_or_ratings(self):
  a=self.m.density_logits(self.c,self.n,self.r,torch.ones(8,dtype=torch.long))[0];b=self.m.density_logits(self.c,self.n,1-self.r,torch.zeros(8,dtype=torch.long))[0];self.assertTrue(torch.equal(a,b))
 def test_query_API_cannot_accept_true_target(self):
  self.assertEqual(list(inspect.signature(self.m.predict_proba).parameters),['c','n','ratings'])
  with self.assertRaises(TypeError):self.m.predict_proba(self.c,self.n,self.r,target=torch.ones(8))
 def test_joint_loss_has_finite_gradients(self):
  self.m.train();loss=self.m.joint_loss(self.c,self.n,self.r,torch.arange(8)%2);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters() if p.grad is not None))
 def test_prediction_batch_invariance(self):
  with torch.no_grad():a=self.m.predict_proba(self.c,self.n,self.r);b=torch.cat([self.m.predict_proba(self.c[k:k+1],self.n[k:k+1],self.r[k:k+1]) for k in range(8)])
  self.assertLess((a-b).abs().max().item(),2e-6)
 def test_encoder_own_training_state_and_unknown_categories(self):
  import numpy as np
  import pandas as pd
  from research_r73_density_model import Encoder,RAW,RATINGS,CATS,NUMS
  X=pd.DataFrame({name:(["A","B","A","B"] if name in CATS else [0,1,2,3]) for name in RAW});X.loc[0,NUMS[-1]]=np.nan;e=Encoder().fit(X);before=[a.copy() for a in [e.median,e.mean,e.scale]];q=X.copy();q[CATS[0]]="UNSEEN";q[NUMS[0]]=100000;c,n,r=e.transform(q);self.assertTrue((c[:,0]==0).all());self.assertEqual(n.shape,(4,8));self.assertTrue(np.isfinite(n).all());self.assertTrue(all(np.array_equal(a,b) for a,b in zip(before,[e.median,e.mean,e.scale])));self.assertEqual(e.fit_rows,4)
  with self.assertRaises(AssertionError):e.transform(X.assign(satisfaction=True))
if __name__=='__main__':unittest.main()

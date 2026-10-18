"""Meaningful label-isolation checks before using synthetic-target neighborhood features."""
import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from research_w3_residual_kernel_features import ResidualKernel
def test_self_label_isolation_and_query_guard():
 raw=pd.DataFrame({'Age':[20,21,22,23,24,25,26,27,28,29],'Flight Distance':[100]*10,'Inflight wifi service':[1,2,3,4,5,1,2,3,4,5],'Gender':['F','M']*5});y=np.array([0,0,1,1,1,0,1,0,1,0]);q=np.linspace(.1,.9,10);splits=list(KFold(2,shuffle=True,random_state=42).split(raw));base=np.empty((10,4))
 for fi,vi in splits:base[vi]=ResidualKernel(raw.iloc[fi],y[fi],q[fi],fi).transform(raw.iloc[vi],vi)
 for j in range(10):
  changed=y.copy();changed[j]=1-changed[j];fi,vi=next((a,b) for a,b in splits if j in b);p=ResidualKernel(raw.iloc[fi],changed[fi],q[fi],fi).transform(raw.iloc[vi],vi);assert np.array_equal(p[np.flatnonzero(vi==j)[0]],base[j]),'Own target leaks into its OOF feature'
 model=ResidualKernel(raw.iloc[:5],y[:5],q[:5],np.arange(5))
 try:model.transform(raw.iloc[[0]],np.array([0]))
 except AssertionError:pass
 else:raise AssertionError('Self query was accepted')
 held=raw.iloc[[6,7]].copy();p=model.transform(held,np.array([100,101]));assert np.array_equal(p,model.transform(held,np.array([102,103]))),'Audit row identity became a predictor'
 print('PASS: 10 target perturbations preserve each own OOF feature; self-query rejected; query labels are not accepted; row identity has no numeric effect')
def test_local_residual_and_shrinkage():
 raw=pd.DataFrame({'Age':[20,20],'Flight Distance':[100,200],'Inflight wifi service':[3,3],'Gender':['F','F']});m=ResidualKernel(raw,np.array([1,0]),np.zeros(2),np.array([0,1]));query=pd.concat([raw,raw.iloc[[0]].assign(**{'Flight Distance':300})],ignore_index=True);p=m.transform(query,np.array([10,11,12]));assert np.allclose(p[:,:2],np.array([[11/21,11/21],[10/21,10/21],[.5,.5]]),atol=1e-14);assert np.array_equal(p[:,2],np.array([0,0,1]));assert np.allclose(p[:,3],np.array([np.log(2),np.log(2),0]));print('PASS: analytic local residual/shrinkage and unseen-distance fallback')
if __name__=='__main__':test_self_label_isolation_and_query_guard();test_local_residual_and_shrinkage()

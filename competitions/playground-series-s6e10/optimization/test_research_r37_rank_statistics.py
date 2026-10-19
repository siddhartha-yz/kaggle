"""Check information not identifiable from prior moments/surprisal/entropy and missing semantics."""
import numpy as np
from research_r37_rank_statistics import conditional_normal_rank
# Same expectedrating, observedclassprobability, entropy ANDvariance; oppositeordinalrank.
p=np.array([[0,.1,.3,.3,.1,.2],[0,.2,.1,.3,.3,.1]])
for value in [p@np.arange(6),p[:,3],-(p[:,1:]*np.log(p[:,1:])).sum(1),p@(np.arange(6)**2)]:assert np.allclose(value[0],value[1])
z=conditional_normal_rank(p,np.arange(6),[3,3]);assert z[0]>0 and z[1]<0 and abs(z[0]+z[1])<1e-7
# Class order is explicit and absentclass0 does not change the positive-rating distribution.
q=p[:,[5,2,1,4,3]];assert np.array_equal(z,conditional_normal_rank(q,[5,2,1,4,3],[3,3]))
assert np.array_equal(conditional_normal_rank(p,np.arange(6),[0,0]),[0,0])
assert conditional_normal_rank([[0,0,0,1,0,0]],np.arange(6),[3])[0]==0
assert conditional_normal_rank([[1,0,0,0,0,0]],np.arange(6),[4])[0]==0
u=np.array([[0,.2,.2,.2,.2,.2]]*2);v=conditional_normal_rank(u,np.arange(6),[1,5]);assert v[0]<0 and v[1]>0 and abs(v[0]+v[1])<1e-7
print('PASS: moment/surprisal/entropy nonidentifiability; arbitrary class order; missing and absent/degenerate classes; signed ordinal tails')

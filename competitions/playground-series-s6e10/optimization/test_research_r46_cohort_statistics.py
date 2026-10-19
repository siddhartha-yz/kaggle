"""Observed-class mapping for both six-class ratings and baggage without class zero."""
import numpy as np
from research_r46_auxiliary import statistics
def main():
    p=np.array([[.1,.2,.3,.1,.2,.1],[.15,.15,.1,.2,.2,.2]])
    ev,s=statistics(p,np.arange(6),np.array([0,5]))
    np.testing.assert_allclose(ev,p@np.arange(6),rtol=1e-6)
    np.testing.assert_allclose(s,-np.log([.1,.2]),rtol=1e-6)
    q=p[:,1:];q=q/q.sum(1)[:,None]
    ev,s=statistics(q,np.arange(1,6),np.array([1,5]))
    np.testing.assert_allclose(ev,q@np.arange(1,6),rtol=1e-6)
    np.testing.assert_allclose(s,-np.log(q[[0,1],[0,4]]),rtol=1e-6)
    try:statistics(q,np.arange(1,6),np.array([0,5]))
    except AssertionError:pass
    else:raise AssertionError('unobserved class must not silently select a different class')
    print('PASS actual-class mapping, expectations, surprisal, absent-class rejection')
if __name__=='__main__':main()

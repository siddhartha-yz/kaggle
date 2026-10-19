import unittest,numpy as np
from research_r14_log_pool import pool,independent_pool,EPS
class LogPoolTest(unittest.TestCase):
    def test_formula_symmetry_identity_and_boundary(self):
        x=np.array([0,.01,.2,.5,.8,.99,1.]);y=x[::-1]
        np.testing.assert_allclose(pool(x,y),independent_pool(x,y),rtol=0,atol=1e-14)
        np.testing.assert_allclose(pool(x,y),.5,rtol=0,atol=1e-11)
        np.testing.assert_allclose(pool(x,x),np.clip(x,EPS,1-EPS),rtol=0,atol=1e-14)
        np.testing.assert_array_equal(pool(x,y),pool(y,x))
        self.assertTrue(np.isfinite(pool(x,y)).all())
if __name__=='__main__':unittest.main()

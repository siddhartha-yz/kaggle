import unittest,numpy as np
from research_r25_posterior import posterior_features
class FullPosteriorTests(unittest.TestCase):
    def test_retains_information_beyond_mean_selfprob_and_entropy(self):
        p=np.array([[.1,.25,.4,.05,.2,0],[.2,.05,.4,.25,.1,0]])
        self.assertAlmostEqual(float((p[0]*np.arange(6)).sum()),float((p[1]*np.arange(6)).sum()))
        self.assertEqual(p[0,2],p[1,2]);self.assertAlmostEqual(float((-p[0]*np.log(np.clip(p[0],1e-300,1))).sum()),float((-p[1]*np.log(np.clip(p[1],1e-300,1))).sum()))
        q=posterior_features(p,np.arange(6));self.assertFalse(np.array_equal(q[0],q[1]))
    def test_class_order_and_missing_states(self):
        p=np.array([[.2,.3,.5]]);a=posterior_features(p,[1,5,0]);b=posterior_features(p[:,[2,0,1]],[0,1,5]);np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a,[[.5,.2,0,0,0,.3]])
    def test_unknown_class_rejected(self):
        with self.assertRaises(AssertionError):posterior_features([[.5,.5]],[0,6])
if __name__=='__main__':unittest.main()

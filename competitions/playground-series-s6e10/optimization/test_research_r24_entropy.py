"""Information content, class-order invariance, and degenerate distributions."""
import unittest,numpy as np
from research_r24_entropy import uncertainty
class ConditionalUncertaintyTests(unittest.TestCase):
    def test_adds_information_beyond_mean_and_observed_probability(self):
        p=np.array([[.25,0,.5,0,.25],[.125,.125,.5,.125,.125]])
        expected,surprisal,entropy,centered=uncertainty(p,np.arange(5),[2,2])
        self.assertEqual(expected[0],expected[1]);self.assertEqual(surprisal[0],surprisal[1])
        self.assertGreater(entropy[1],entropy[0]);self.assertLess(centered[1],centered[0])
    def test_class_permutation_invariance(self):
        p=np.array([[.1,.3,.6],[.6,.4,0]]);order=[2,0,1]
        a=uncertainty(p,[0,2,5],[2,5]);b=uncertainty(p[:,order],np.array([0,2,5])[order],[2,5])
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-6)
    def test_certain_and_unknown_observed_class_are_finite(self):
        e,s,h,c=uncertainty([[1,0],[1,0]],[0,5],[0,2]);self.assertTrue(np.isfinite(c).all());np.testing.assert_array_equal(h,[0,0]);self.assertEqual(s[0],0);self.assertAlmostEqual(float(s[1]),-np.log(1e-6),places=5)
if __name__=='__main__':unittest.main()

import unittest,numpy as np
from research_r21_conditional import conditional_statistics,EPS
class ConditionalTest(unittest.TestCase):
    def test_actual_class_labels_unknown_values_and_permutation(self):
        p=np.array([[.2,.3,.5],[.6,.1,.3],[0,1,0.]])
        classes=np.array([1,2,4]);observed=np.array([2,0,4])
        e,s=conditional_statistics(p,classes,observed)
        np.testing.assert_allclose(s,[-np.log(.3),-np.log(EPS),-np.log(EPS)],rtol=1e-7)
        perm=np.array([2,0,1]);ee,ss=conditional_statistics(p[:,perm],classes[perm],observed)
        np.testing.assert_array_equal(e,ee);np.testing.assert_array_equal(s,ss)
        ee,ss=conditional_statistics(p[::-1],classes,observed[::-1]);np.testing.assert_array_equal(e,ee[::-1]);np.testing.assert_array_equal(s,ss[::-1])
    def test_same_mean_can_hide_observed_probability(self):
        e,s=conditional_statistics([[.5,0,.5],[0,1,0]],[0,2,4],[2,2])
        np.testing.assert_array_equal(e,[2,2]);self.assertGreater(s[0],13);self.assertEqual(s[1],0)
    def test_invalid_probability_rejected(self):
        with self.assertRaises(AssertionError):conditional_statistics([[np.nan,1]],[0,1],[0])
        with self.assertRaises(AssertionError):conditional_statistics([[.2,.2]],[0,1],[0])
if __name__=='__main__':unittest.main()

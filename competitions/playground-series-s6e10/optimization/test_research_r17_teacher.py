import unittest,numpy as np,pandas as pd
from research_r17_teacher_features import TeacherFeatures
class TeacherTest(unittest.TestCase):
    def test_clipped_teacher_finite_and_aligned(self):
        out=pd.DataFrame({f'n{j}':np.zeros(7,dtype='float32') for j in range(25)})
        for j in range(22):out[f'c{j}']='x'
        q=np.array([0,1e-6,.1,.5,.8,1-1e-6,1.])
        result=TeacherFeatures.append_teacher(out.copy(),q)
        expected=np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6))).astype('float32')
        np.testing.assert_array_equal(result.original_teacher_logit,expected)
        self.assertEqual(result.original_teacher_logit.iloc[3],0)
        reverse=TeacherFeatures.append_teacher(out.iloc[::-1].copy(),q[::-1])
        np.testing.assert_array_equal(reverse.original_teacher_logit.iloc[::-1],result.original_teacher_logit)
        with self.assertRaises(AssertionError):TeacherFeatures.append_teacher(out.copy(),np.full(7,np.nan))
if __name__=='__main__':unittest.main()

import unittest
import numpy as np,pandas as pd
from research_original_context import OriginalContext
class ContextTests(unittest.TestCase):
 def test_source_mean_deviation_and_unseen_fallback(self):
  x=pd.DataFrame({'Flight Distance':[10,10,20,20],'Age':[20.,40.,50.,70.],'Gender':['F','M','F','M']});t=OriginalContext().fit(x);before=t.mapping.copy();v=pd.DataFrame({'Flight Distance':[10,999],'Age':[30.,80.],'Gender':['F','M']});r=t.transform(v);expected=(60+20*45)/22
  np.testing.assert_allclose(r.orig_ctx_mean_00,[expected,45]);np.testing.assert_allclose(r.orig_ctx_delta_00,[30-expected,35]);pd.testing.assert_frame_equal(before,t.mapping)
 def test_labels_and_ids_rejected(self):
  for c in ['id','satisfaction']:
   x=pd.DataFrame({'Flight Distance':[10,20],'Age':[20.,40.],c:[0,1]})
   with self.assertRaises(AssertionError):OriginalContext().fit(x)
if __name__=='__main__':unittest.main()

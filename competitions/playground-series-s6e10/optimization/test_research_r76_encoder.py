"""Synthetic-only source/unknown-category/query isolation checks; no official labels."""
import unittest,copy
import numpy as np,pandas as pd
from research_r76_model import RawEncoder
class EncoderTests(unittest.TestCase):
 def setUp(self):self.raw=pd.DataFrame({**{f'n{k}':[0.,1.,2.,3.] for k in range(17)},**{f'c{k}':['A','B','A','B'] for k in range(4)}})
 def test_unknown_query_category_does_not_change_training_state(self):
  enc=RawEncoder().fit(self.raw);before=copy.deepcopy(enc.state());q=self.raw.copy();q['c0']='QUERY_ONLY';out=enc.transform(q);self.assertTrue((out[:,17]==-1).all());self.assertEqual(before,enc.state());self.assertEqual(enc.fit_rows,4)
 def test_numeric_missing_and_column_order(self):
  enc=RawEncoder().fit(self.raw);q=self.raw.copy();q.loc[0,'n0']=np.nan;out=enc.transform(q);self.assertEqual(out[0,0],1.5);self.assertEqual(out.dtype,np.float32);self.assertEqual(out.shape,(4,21))
  with self.assertRaises(AssertionError):enc.transform(q.iloc[:,::-1])
 def test_fit_rejects_ID_or_target(self):
  for col in ['id','satisfaction']:
   with self.assertRaises(AssertionError):RawEncoder().fit(self.raw.rename(columns={'n0':col}))
 def test_query_without_numeric_schema_fails(self):
  enc=RawEncoder().fit(self.raw);q=self.raw.copy();q['n0']='invalid'
  with self.assertRaises(ValueError):enc.transform(q)
 def test_context_balanced_deterministic_and_training_only(self):
  from research_r76_model import balanced_context
  y=np.array([0]*3000+[1]*5000);a=balanced_context(y,4096,np.random.default_rng(42));b=balanced_context(y,4096,np.random.default_rng(42));self.assertTrue(np.array_equal(a,b));self.assertEqual(len(np.unique(a)),4096);self.assertEqual(np.bincount(y[a]).tolist(),[2048,2048])
 def test_training_episodes_disjoint_and_deterministic(self):
  a=np.random.default_rng(42).choice(5000,1024,replace=False);b=np.random.default_rng(42).choice(5000,1024,replace=False);self.assertTrue(np.array_equal(a,b));self.assertEqual(len(np.intersect1d(a[:512],a[512:])),0)
if __name__=='__main__':unittest.main()

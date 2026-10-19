"""Synthetic generative, label-isolation and inference invariance checks."""
import unittest
import numpy as np
import pandas as pd
from research_r9_linkage_features import ExemplarLinkage, FD

class LinkageTest(unittest.TestCase):
    def test_label_free_fit_and_posterior(self):
        rng = np.random.default_rng(7)
        source = pd.DataFrame({FD:np.ones(8,dtype='int64'),
                              **{f'field{j}':np.roll(np.arange(8),j) for j in range(20)}})
        selected = rng.integers(0,8,1024)
        raw = source.iloc[selected].reset_index(drop=True).copy()
        noise = rng.random((len(raw),20)) < .2
        values = raw.drop(columns=FD).to_numpy()
        values[noise] = rng.integers(0,8,noise.sum())
        raw.iloc[:,1:] = values
        model = ExemplarLinkage(sample_rows=1024,iterations=10).fit(raw,source)
        self.assertFalse(model.fit_summary['source_target_used_for_EM'])
        self.assertFalse(model.fit_summary['official_target_used'])
        self.assertGreater(min(np.diff([r['penalized_log_likelihood'] for r in model.trace])),-1e-4)
        labels = np.arange(8)%2
        a = model.transform(source,labels)
        self.assertLess(np.max(np.abs(a.link_teacher-labels)),.001)
        flipped = model.transform(source,1-labels)
        self.assertLess(np.max(np.abs(a.link_teacher+flipped.link_teacher-1)),1e-7)
        self.assertLess(np.max(np.abs(a.link_conditional_teacher+flipped.link_conditional_teacher-1)),1e-7)
        np.testing.assert_array_equal(a.iloc[:,2:],flipped.iloc[:,2:])
        reverse = model.transform(source.iloc[::-1],labels).iloc[::-1].reset_index(drop=True)
        np.testing.assert_array_equal(a,reverse)
        np.testing.assert_array_equal(a,model.transform(source,labels,batch_size=3))
        unknown = source.iloc[:1].copy()
        unknown[FD] = 999
        prediction = model.transform(unknown,labels)
        self.assertEqual(prediction.link_teacher.iloc[0],.5)
        self.assertEqual(prediction.link_probability.iloc[0],0)

if __name__ == '__main__': unittest.main()

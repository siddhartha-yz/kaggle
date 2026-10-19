import unittest,numpy as np,pandas as pd
from research_r15_segment_features import SegmentPrior,crossfit
class SegmentTest(unittest.TestCase):
    def test_heldout_labels_are_excluded_and_unknown_falls_back(self):
        n=50;raw=pd.DataFrame({'Class':['Eco','Biz']*(n//2),'Type of Travel':['B','P']*(n//2),
            'Customer Type':['L','L','D','D','L']*(n//5)})
        y=np.arange(n)%3==0
        out,fold=crossfit(raw,y);held=fold==0;changed=y.copy();changed[held]=~changed[held]
        again,newfold=crossfit(raw,changed)
        np.testing.assert_array_equal(fold,newfold)
        np.testing.assert_array_equal(out[held],again[held])
        model=SegmentPrior().fit(raw,y);query=raw.iloc[:1].copy();query['Class']='UNKNOWN'
        self.assertAlmostEqual(float(model.transform(query)[0]),float(y.mean()),places=7)
        reverse=model.transform(raw.iloc[::-1])[::-1]
        np.testing.assert_array_equal(reverse,model.transform(raw))
        self.assertEqual(int(model.table['count'].sum()),n)
if __name__=='__main__':unittest.main()

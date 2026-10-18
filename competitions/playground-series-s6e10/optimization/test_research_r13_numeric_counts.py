import unittest
import numpy as np,pandas as pd
from research_r13_numeric_counts import NumericCountFeatures,COUNT_COLUMNS
from research_next_auto_te_features import RATINGS

class NumericCountsTest(unittest.TestCase):
    def test_exact_query_exclusion_and_missing(self):
        rng=np.random.default_rng(43);n=40
        data={c:rng.integers(0,6,n) for c in RATINGS}
        data.update({'Age':np.arange(n)%7+20,'Flight Distance':np.arange(n)%5+100,
            'Departure Delay in Minutes':np.arange(n)%3,'Arrival Delay in Minutes':(np.arange(n)%4).astype(float),
            'Gender':['F','M']*(n//2),'Customer Type':['L','D']*(n//2),
            'Type of Travel':['B','P']*(n//2),'Class':['Eco','Biz']*(n//2)})
        raw=pd.DataFrame(data);raw.loc[[0,4],'Arrival Delay in Minutes']=np.nan;y=np.arange(n)%2
        tr=NumericCountFeatures();train=tr.fit_transform(raw,y);valid=tr.transform(raw)
        for j,c in enumerate(COUNT_COLUMNS):
            expected=raw[c].map(raw[c].value_counts(dropna=False)).to_numpy('float32')
            np.testing.assert_array_equal(train[f'numeric_frequency_{j}'],expected-1)
            np.testing.assert_array_equal(valid[f'numeric_frequency_{j}'],expected)
        self.assertEqual(train.numeric_frequency_3.iloc[0],1)
        query=raw.iloc[:1].copy();query['Flight Distance']=999
        self.assertEqual(tr.transform(query).numeric_frequency_1.iloc[0],0)
        reverse=tr.transform(raw.iloc[::-1])
        np.testing.assert_array_equal(reverse.numeric_frequency_0.iloc[::-1],valid.numeric_frequency_0)
        tr2=NumericCountFeatures();changed=tr2.fit_transform(raw,1-y)
        np.testing.assert_array_equal(train.iloc[:,-4:],changed.iloc[:,-4:])
if __name__=='__main__':unittest.main()

"""LOO equals physically removing the query; missing values and batch invariance."""
import unittest
import numpy as np,pandas as pd
from research_features import DistanceFeatures
from research_r12_population_features import PopulationDistance

class PopulationTest(unittest.TestCase):
    def test_leave_query_out_matches_explicit_refit(self):
        data={c:[1,3,2,5] for c in ['Age','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness','Departure Delay in Minutes']}
        data.update({'Arrival Delay in Minutes':[np.nan,4.,2.,8.],'Flight Distance':[10,10,20,10],
            'Gender':['M','F','M','F'],'Customer Type':['L','D','L','D'],
            'Type of Travel':['B','P','B','P'],'Class':['Eco','Eco','Biz','Biz']})
        pool=pd.DataFrame(data);train=pool.iloc[:3].copy();reference=pool.iloc[3:].copy()
        model=PopulationDistance().fit_population(train,reference)
        result=model.transform_population(pool,included=True)
        numeric=[model.names[c] for c in model.output_columns if c.startswith('distance_')]
        for i in range(len(pool)):
            removed=pool.drop(index=i).reset_index(drop=True)
            explicit=DistanceFeatures('distance_stats_numeric').fit(removed)
            named=explicit._raw(pool.iloc[i:i+1])
            expected=named[[c for c in model.output_columns if c.startswith('distance_')]].to_numpy('float64')
            np.testing.assert_allclose(result.iloc[i:i+1][numeric].to_numpy('float64'),expected,rtol=0,atol=1e-12)
        reverse=model.transform_population(pool.iloc[::-1],included=True).iloc[::-1]
        pd.testing.assert_frame_equal(result,reverse)
        external=pool.iloc[:1].copy();external['Flight Distance']=999
        pred=model.transform_population(external,included=False)
        self.assertEqual(pred[model.names['distance_count']].iloc[0],0)
        self.assertEqual(pred[model.names['distance_mean_Age']].iloc[0],pool.Age.mean())
        with self.assertRaises(AssertionError):model.transform_population(external,included=True)

if __name__=='__main__':unittest.main()

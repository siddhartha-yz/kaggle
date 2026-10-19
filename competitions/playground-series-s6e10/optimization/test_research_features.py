"""Leakage and outer-train mapping tests for target-aware feature code."""
import unittest
import numpy as np,pandas as pd
from research_features import DistanceFeatures,RATING_CATS
class FeatureTests(unittest.TestCase):
 def frame(self):
  x=pd.DataFrame({'Age':[20,22,30,32,40,42,50,52],'Flight Distance':[10,10,20,20,30,30,40,40],'Gender':['F','M']*4,'Class':['Eco','Biz']*4})
  for c in RATING_CATS:x[c]=[0,1,2,3,4,5,1,2]
  return x
 def test_outer_only_frequency_and_unseen_fallback(self):
  x=self.frame();t=DistanceFeatures('distance_group_stats').fit(x)
  v=x.iloc[:2].copy();v.loc[v.index[0],'Flight Distance']=999
  out=t.transform(v);names=t.names
  self.assertEqual(out.iloc[0][names['distance_count']],0)
  self.assertEqual(out.iloc[1][names['distance_count']],2)
  self.assertAlmostEqual(out.iloc[0][names['distance_mean_Age']],x.Age.mean())
  self.assertEqual(int(t.counts.sum()),len(x))
 def test_training_te_excludes_own_label_and_inner_prior(self):
  x=self.frame();y=np.array([0,1]*4);ev=np.array([0,2,4,6]);od=np.array([1,3,5,7]);folds=[(od,ev),(ev,od)]
  t=DistanceFeatures('distance_te',[['Flight Distance']],alpha=100)
  out=t.fit_transform_te(x,y,folds=folds)
  np.testing.assert_array_equal(out['te00'],1-y)
  changed=y.copy();changed[0]=1
  t2=DistanceFeatures('distance_te',[['Flight Distance']],alpha=100)
  out2=t2.fit_transform_te(x,changed,folds=folds)
  self.assertEqual(out.iloc[0]['te00'],out2.iloc[0]['te00'])
  v=x.iloc[:1].copy();v['Flight Distance']=999
  self.assertAlmostEqual(t.transform_te(v).iloc[0]['te00'],y.mean())
 def test_outer_valid_values_do_not_change_fitted_maps(self):
  x=self.frame();t=DistanceFeatures('distance_group_stats').fit(x)
  before=t.counts.copy();v=x.copy();v['Age']=10000;v['Flight Distance']=999;t.transform(v)
  pd.testing.assert_series_equal(before,t.counts)
 def test_target_and_id_are_rejected(self):
  for column in ['id','satisfaction']:
   x=self.frame();x[column]=0
   with self.assertRaises(AssertionError):DistanceFeatures().fit(x)
 def test_rating_distributions_are_normalized_and_train_only(self):
  from research_distributions import DistanceDistributions
  x=self.frame();t=DistanceDistributions().fit(x);v=x.iloc[:2].copy();v['Flight Distance']=[10,999]
  prior=dict(t.rating_prop_prior);maps={k:m.copy() for k,m in t.rating_prop_maps.items()};out=t.transform(v)
  for c in RATING_CATS:
   cols=[t.names[f'distance_rating_prop_{c}_{n}'] for n in range(6)]
   np.testing.assert_allclose(out[cols].sum(axis=1),1,rtol=0,atol=1e-14)
   np.testing.assert_allclose(out.iloc[1][cols].to_numpy(dtype=float),[(x[c]==n).mean() for n in range(6)])
  v['Age']=9999;t.transform(v)
  self.assertEqual(prior,t.rating_prop_prior)
  for k,m in maps.items():pd.testing.assert_series_equal(m,t.rating_prop_maps[k])
 def test_interaction_encoding_excludes_entire_inner_holdout(self):
  x=self.frame();y=np.array([0,1]*4);a=np.arange(4);b=np.arange(4,8);folds=[(b,a),(a,b)]
  keys=[['Flight Distance','Class'],['Age','Class']]
  t=DistanceFeatures('distance_stats_numeric',keys,alpha=20);before=t.fit_transform_te(x,y,folds=folds)
  changed=y.copy();changed[a]=1-changed[a]
  u=DistanceFeatures('distance_stats_numeric',keys,alpha=20);after=u.fit_transform_te(x,changed,folds=folds)
  np.testing.assert_array_equal(before.iloc[a][['te00','te01']],after.iloc[a][['te00','te01']])
  v=x.iloc[:2].copy();v['Flight Distance']=999;v['Age']=999
  np.testing.assert_allclose(t.transform_te(v)[['te00','te01']],y.mean())
if __name__=='__main__':unittest.main()

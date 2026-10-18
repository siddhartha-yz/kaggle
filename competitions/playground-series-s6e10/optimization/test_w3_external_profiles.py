"""Check singleton groups, missing values, self-subtraction and row-identity protection."""
import pandas as pd,numpy as np
from research_w3_external_features import ExternalProfiles
def main():
 raw=pd.DataFrame({'Flight Distance':[10,10,10,10,20,20,20,30],'Age':[20,21,22,23,30,31,32,44],'Arrival Delay in Minutes':[np.nan,1,2,3,4,np.nan,6,8],'Gender':['M','F','M','F','M','F','M','F']});ids=np.arange(100,108);p=ExternalProfiles().fit(raw,ids);got=p.transform(raw,ids);maximum=0.
 for i in range(len(raw)):
  keep=np.arange(len(raw))!=i;independent=ExternalProfiles().fit(raw.loc[keep].reset_index(drop=True),ids[keep]);expected=independent.transform(raw.iloc[[i]]);assert list(expected)==list(got);maximum=max(maximum,float(np.max(np.abs(expected.to_numpy()[0]-got.iloc[i].to_numpy()))));np.testing.assert_allclose(expected.to_numpy()[0],got.iloc[i].to_numpy(),rtol=0,atol=1e-6)
 bad=raw.iloc[:1].copy();bad.Age=999
 try:p.transform(bad,ids[:1])
 except AssertionError:pass
 else:raise AssertionError('Rowidentityguarddidnotrejectchangedsourcefeatures')
 print({'status':'passed','rows_bruteforce_checked':8,'maximum_self_subtraction_difference':maximum,'missing_values_and_singleton_group':True,'row_identity_guard':True})
if __name__=='__main__':main()

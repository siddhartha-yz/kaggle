"""13 changed numeric coordinates, all other 61 fields exact, raw identities preserved."""
import numpy as np,pandas as pd
from research_r47_features import RankCoordinateFeatures,RATINGS
from research_r23_features import ConditionalFeatures
def main():
    n=30;t=np.arange(n);raw=pd.DataFrame({'Age':20+t%7,'Flight Distance':100+t%3})
    for j,c in enumerate(RATINGS):raw[c]=(t+j)%6
    raw['Departure Delay in Minutes']=t%4;raw['Arrival Delay in Minutes']=t%5
    for c in ['Gender','Customer Type','Type of Travel','Class']:raw[c]=np.where(t%2,'a','b')
    aux=pd.DataFrame({f'aux{j}':np.zeros(n,dtype='float32')+j/100 for j in range(27)})
    for j in range(13):aux[f'conditional_positive_rank_{j}']=np.linspace(-2,2,n,dtype='float32')+j/10
    y=t%2;base=ConditionalFeatures();B=base.fit_transform(raw,y,aux.iloc[:,:27]);tr=RankCoordinateFeatures();A=tr.fit_transform(raw,y,aux);pd.testing.assert_frame_equal(A.drop(columns=RATINGS),B.drop(columns=RATINGS),check_exact=True)
    V=tr.transform(raw,aux);W=base.transform(raw,aux.iloc[:,:27]);pd.testing.assert_frame_equal(V.drop(columns=RATINGS),W.drop(columns=RATINGS),check_exact=True)
    assert A.shape==B.shape==(n,74)
    for j,c in enumerate(RATINGS):
        identity=f'value_cat_{tr.numeric.index(c)}';assert np.array_equal(A[identity].astype('int64'),raw[c]) and np.array_equal(A[c],aux.iloc[:,27+j])
    print('PASS unchanged61fields, native74geometry, exactrawratingidentity, newrankcoordinates')
if __name__=='__main__':main()

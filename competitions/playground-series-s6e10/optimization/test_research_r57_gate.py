"""Check analytic gradient, bounded pooling and unknown-cohort fallback without official data."""
import numpy as np,pandas as pd
from research_r57_gate import CohortGate,objective,logits
from scipy.special import expit
def main():
    rng=np.random.default_rng(20261057);base=rng.normal(size=40);delta=rng.normal(size=40);index=np.arange(40)%3;y=np.arange(40)%2;theta=np.array([.1,-.2,.3]);value,g=objective(theta,base,delta,index,y,100.)
    numeric=[]
    for j in range(3):
        d=np.zeros(3);d[j]=1e-5;numeric.append((objective(theta+d,base,delta,index,y,100.)[0]-objective(theta-d,base,delta,index,y,100.)[0])/2e-5)
    error=float(np.max(np.abs(g-numeric)));assert error<1e-6
    raw=pd.DataFrame({'Class':['Eco','Business']*20,'Type of Travel':['Business travel']*40,'Customer Type':['Loyal Customer']*40});p=expit(base);gate=CohortGate().fit(raw,p,p,y);assert np.array_equal(gate.theta,np.zeros(2)) and np.max(np.abs(gate.predict(raw,p,p)-p))<1e-15
    a=expit(base);b=expit(base+delta);model=CohortGate().fit(raw,a,b,y);pred=model.predict(raw,a,b);assert np.isfinite(pred).all() and np.all(pred>=np.minimum(a,b)-1e-15) and np.all(pred<=np.maximum(a,b)+1e-15)
    unknown=raw.copy();unknown['Class']='UNKNOWN';assert np.array_equal(model.coefficients(unknown),np.zeros(40));z,_=logits(a,b);assert np.array_equal(model.predict(unknown,a,b),expit(z));print('R57_SYNTHETIC_GATE_TEST_PASSED',error,flush=True)
if __name__=='__main__':main()

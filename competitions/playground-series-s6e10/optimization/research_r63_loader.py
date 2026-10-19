"""The source-stage predeclared neural view: sourceLLR13 plus oldconditionalSurprisal14."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline
from research_r62_loader import auxiliary as allviews
import numpy as np,pandas as pd
def auxiliary(ids,test=False):
    combined=allviews(ids,test=test);out=pd.concat([combined.iloc[:,27:],combined.iloc[:,13:27]],axis=1);assert out.shape==(len(ids),27) and np.isfinite(out.to_numpy()).all();return out
def load_split(split):
    df,raw,y,splits,_,tree,fold,_=baseline(split);best=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz');assert np.array_equal(best['id'],df.id) and np.array_equal(best['target'],y) and np.array_equal(best['fold'],fold)
    return df,raw,y,splits,best['candidate'],tree,fold,auxiliary(df.id)

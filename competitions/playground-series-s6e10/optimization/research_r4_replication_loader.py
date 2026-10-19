"""Bind a completely refitted baseline to a newly frozen threefold partition."""
from research_w3_common import ROOT,sha
import numpy as np,pandas as pd,json

def load_split(split):
 assert split=='oof';out=ROOT/'research_artifacts/r4_baseline_replication3';rec=json.loads((out/'result.json').read_text());assert rec['status']=='complete' and rec['oof_sha256']==sha(out/'oof.npz') and rec['all_baseline_components_freshly_trained'];df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);z=np.load(out/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fold=z['fold'];assert set(fold)=={0,1,2};splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(3)];ref=z['pred'];old=z['old_nn'];return df,raw,y,splits,ref,old,fold

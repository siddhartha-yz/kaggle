from research_w3_common import ROOT
import sys,json,numpy as np,pandas as pd
STAGE='threefold' if '--threefold' in sys.argv else 'fivefold'
def load_split(split):
 assert split=='oof' and json.loads((ROOT/'research_r6_half_result.json').read_text())['eligible_for_candidate'];df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction']);path='r4_lean_gpu_init_oof' if STAGE=='fivefold' else 'r4_lean_gpu_init_replication3';z=np.load(ROOT/'research_artifacts'/path/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);fold=z['fold'];nn=z['pred'];v5=(z['candidate']-.25*nn)/.75;ref=.5*v5+.5*nn;splits=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in sorted(np.unique(fold))];return df,raw,y,splits,ref,nn,fold

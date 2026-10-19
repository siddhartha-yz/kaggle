"""One externally specified 50/50 reliability control; never a blend-weight search."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,pandas as pd

def main():
 df=pd.read_csv(ROOT/'data/train.csv');y=df.satisfaction.to_numpy('int8');result={'status':'complete','updated_utc':now(),'stages':{},'weights':{'v5':.5,'lean_nn8':.5},'publicscore_used_for_weight':False}
 for stage,path in [('fivefold','r4_lean_gpu_init_oof'),('threefold','r4_lean_gpu_init_replication3')]:
  z=np.load(ROOT/'research_artifacts'/path/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);ref=z['candidate'];nn=z['pred'];v5=(ref-.25*nn)/.75;assert v5.min()>-1e-12 and v5.max()<1+1e-12;candidate=.5*v5+.5*nn;record=evidence(y,ref,candidate,z['fold']);record['reference_sha256']=sha(ROOT/'research_artifacts'/path/'oof.npz');result['stages'][stage]=record
 result['eligible_for_candidate']=all(x['eligible_for_new_partition_replication'] for x in result['stages'].values());result['formal_used']=1;result['formal_remaining']=1;result['actualTop100verified']=False;dump(ROOT/'research_r6_half_result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

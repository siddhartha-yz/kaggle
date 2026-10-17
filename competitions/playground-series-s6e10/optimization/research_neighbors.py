"""Original-only nearest-neighbor evidence within exact Flight Distance groups."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd
from research_original import safe_original
ROOT=Path(__file__).resolve().parent
NAMES=['orig_nn_y1','orig_nn_y3','orig_nn_y10','orig_nn_weighted_y10','orig_nn_distance1','orig_nn_mean_distance3','orig_nn_mean_distance10','orig_nn_label_std10']
class OriginalNeighbors:
 def __init__(self,X,y):
  assert 'id' not in X and 'satisfaction' not in X
  self.columns=list(X);self.cats=X.select_dtypes(['object','string']).columns.tolist();self.numeric=[c for c in X if c not in self.cats and c!='Flight Distance'];self.y=np.asarray(y,dtype=float);self.prior=float(self.y.mean());self.source=X.reset_index(drop=True);self.num=self._numeric(self.source);self.cat=self.source[self.cats].fillna('__MISSING__').astype(str).to_numpy();self.groups=self.source.groupby('Flight Distance',sort=False).indices
 def _numeric(self,X):
  out=[]
  for c in self.numeric:
   a=X[c].to_numpy(dtype=float)
   if c=='Age':a=a/100.
   elif c in ['Arrival Delay in Minutes','Departure Delay in Minutes']:a=np.log1p(np.maximum(a,0))/np.log1p(360.)
   else:a=a/5.
   out.append(np.where(np.isfinite(a),a,-1.))
  return np.column_stack(out)
 def transform(self,X):
  assert 'id' not in X and 'satisfaction' not in X and list(X)==self.columns
  X=X.reset_index(drop=True);num=self._numeric(X);cat=X[self.cats].fillna('__MISSING__').astype(str).to_numpy();out=np.empty((len(X),8),dtype=float);out[:,:4]=self.prior;out[:,4:7]=1.;out[:,7]=0.;matched=np.zeros(len(X),dtype=bool)
  groups=X.groupby('Flight Distance',sort=False).indices
  for g,(distance,qi) in enumerate(groups.items()):
   oi=self.groups.get(distance)
   if oi is None:continue
   matched[qi]=True;on=self.num[oi];oc=self.cat[oi];oy=self.y[oi];k=min(10,len(oi))
   for begin in range(0,len(qi),128):
    idx=qi[begin:begin+128];d=np.zeros((len(idx),len(oi)),dtype=float)
    for j in range(len(self.numeric)):d+=np.abs(num[idx,j,None]-on[None,:,j])
    for j in range(len(self.cats)):d+=(cat[idx,j,None]!=oc[None,:,j])
    d/=len(self.numeric)+len(self.cats)
    # Stable full argsort makes ties deterministic in original source row order.
    near=np.argsort(d,axis=1,kind='stable')[:,:k];nd=np.take_along_axis(d,near,axis=1);ny=oy[near];weight=np.exp(-nd/.05);k3=min(3,k)
    out[idx,0]=ny[:,0];out[idx,1]=ny[:,:k3].mean(axis=1);out[idx,2]=ny.mean(axis=1);out[idx,3]=(weight*ny).sum(axis=1)/weight.sum(axis=1);out[idx,4]=nd[:,0];out[idx,5]=nd[:,:k3].mean(axis=1);out[idx,6]=nd.mean(axis=1);out[idx,7]=ny.std(axis=1)
   if g%500==0:print('distance_groups',g,'of',len(groups),flush=True)
  assert np.isfinite(out).all();return pd.DataFrame(out,columns=NAMES),matched
def main():
 out=ROOT/'research_artifacts/original_neighbors_v1';out.mkdir(exist_ok=False);start=time.monotonic();original,train,test,cols,audit=safe_original();model=OriginalNeighbors(original[cols],original.satisfaction.astype('int8'))
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'hypothesis':'An original-only local neighborhood may preserve generative similarity missed by a global original teacher. It is a model feature, never a replacement for synthetic labels.','distance':'Exact Flight Distance group, then average normalized L1 over16 numerical and Hamming over4 string predictors. Age/100, ratings/5, log1p delays/log361; missing=-1. Equal feature weights, no synthetic-label fitting.','features':NAMES,'top_k':[1,3,10],'weight_temperature':.05,'source_audit':audit,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'synthetic_targets_used':False};(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 result={}
 for name,df in [('train',train),('test',test)]:
  features,matched=model.transform(df[cols]);zeros=int(((features.orig_nn_distance1==0)&matched).sum());assert zeros==0,'Unexpected exact match after global exclusion'
  np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),columns=np.asarray(NAMES,dtype=str),values=features.to_numpy());result[name]={'rows':len(df),'same_distance_source_rows_available':int(matched.sum()),'zero_distance_matches':zeros,'minimum_matched_distance':float(features.orig_nn_distance1[matched].min())}
 result.update(status='complete',elapsed_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat());(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

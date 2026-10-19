"""Source-only unsupervised Distance means and raw-minus-source-mean features."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd,joblib
from research_original import safe_original
ROOT=Path(__file__).resolve().parent
class OriginalContext:
 def fit(self,X):
  assert 'id' not in X and 'satisfaction' not in X
  cats=X.select_dtypes(['object','string']).columns.tolist();self.numeric=[c for c in X if c not in cats and c!='Flight Distance'];self.priors=X[self.numeric].mean();g=X.groupby('Flight Distance',sort=False);self.mapping=(g[self.numeric].sum()+20*self.priors)/(g[self.numeric].count()+20);return self
 def transform(self,X):
  result={}
  for i,c in enumerate(self.numeric):
   mean=X['Flight Distance'].map(self.mapping[c]).fillna(self.priors[c]);result[f'orig_ctx_mean_{i:02}']=mean.to_numpy();result[f'orig_ctx_delta_{i:02}']=X[c].to_numpy()-mean.to_numpy()
  return pd.DataFrame(result)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((ROOT/'research_stage10_export_state.json').read_text())['status']=='complete'
 out=ROOT/'research_artifacts/original_context_v1';out.mkdir(exist_ok=False);start=time.monotonic();original,train,test,cols,audit=safe_original()
 from research_original_context import OriginalContext as StableOriginalContext
 tr=StableOriginalContext().fit(original[cols]);joblib.dump(tr,out/'transformer.joblib')
 for label,df in [('train',train),('test',test)]:
  result=tr.transform(df[cols]);assert not np.isinf(result.to_numpy()).any();np.savez_compressed(out/f'{label}_features.npz',id=df.id.to_numpy(),columns=np.asarray(result.columns,dtype=str),values=result.to_numpy())
  restored=joblib.load(out/'transformer.joblib');np.testing.assert_allclose(result.to_numpy(),restored.transform(df[cols]).to_numpy(),rtol=0,atol=0,equal_nan=True)
 record={'status':'complete','source_audit':audit,'source_rows':len(original),'numeric_columns':tr.numeric,'shrinkage':20,'features':list(result),'synthetic_labels_used':False,'original_labels_used':False,'source_only_fit':True,'source_groups':len(tr.mapping),'transformer_sha256':sha(out/'transformer.joblib'),'runner_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-start,'completed_utc':datetime.now(timezone.utc).isoformat()};(out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()

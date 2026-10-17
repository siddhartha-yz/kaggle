"""A second independent original-only teacher; public reference read, not executed."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,signal
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from research_original import safe_original
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(X,vocab=None):
 if vocab is None:vocab={c:{v:i for i,v in enumerate(sorted(X[c].fillna('__MISSING__').astype(str).unique()))} for c in X.select_dtypes(['object','string'])}
 out=X.copy()
 for c,m in vocab.items():out[c]=out[c].fillna('__MISSING__').astype(str).map(m).fillna(-1)
 return out.astype('float32'),vocab
def main():
 p=json.loads((ROOT/'research_protocol.json').read_text());assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 def stop(s,f):raise TimeoutError('Hard cutoff')
 signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int((datetime.fromisoformat(p['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())))
 out=ROOT/'research_artifacts/original_xgb_v1';out.mkdir(exist_ok=False);start=time.monotonic();original,train,test,cols,audit=safe_original();raw=original[cols];y=original.satisfaction.astype('int8').to_numpy();split=np.load(ROOT/'research_artifacts/original_teacher_v1/original_selection_split.npz');fi,vi=split['train'],split['valid']
 params=dict(n_estimators=600,learning_rate=.05,max_depth=8,min_child_weight=1,reg_lambda=1,subsample=.8,colsample_bytree=.5,max_bin=256,objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cpu',n_jobs=6,random_state=20261018)
 plan={'params':params,'fixed_iterations':True,'synthetic_targets_used_for_training_or_selection':False,'source_audit':audit,'reference':'https://www.kaggle.com/code/goodpjw2008/s6e10-two-view-gbdt-realmlp-stack-lb-0-96125','reference_handling':'Read-only algorithm inspection. This implementation uses our audited source exclusion and CPU limits; notebook code not executed.','runner_sha256':sha(__file__),'frozen_utc':datetime.now(timezone.utc).isoformat()};(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 X,vocab=matrix(raw.iloc[fi]);V,_=matrix(raw.iloc[vi],vocab);model=XGBClassifier(**params);model.fit(X,y[fi]);selection_auc=float(roc_auc_score(y[vi],model.predict_proba(V)[:,1]));model.save_model(str(out/'selection_model.json'))
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc']);X,vocab=matrix(raw);model=XGBClassifier(**params);model.fit(X,y);model.save_model(str(out/'model.json'));(out/'vocabulary.json').write_text(json.dumps(vocab,indent=2)+'\n')
 for name,df in [('train',train),('test',test)]:
  M,_=matrix(df[cols],vocab);pred=model.predict_proba(M)[:,1];assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),columns=np.asarray(['orig_teacher_xgb'],dtype=str),values=pred[:,None])
 result={'status':'complete','source_rows':len(original),'original_validation_auc':selection_auc,'model_sha256':sha(out/'model.json'),'vocabulary_sha256':sha(out/'vocabulary.json'),'elapsed_seconds':time.monotonic()-start,'ended_utc':datetime.now(timezone.utc).isoformat()};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

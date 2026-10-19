"""Recreate a frozen ensemble submission from saved models and official data."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from iterate import features,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='best_ensemble.json');p.add_argument('--output',default='reproduced_submission.csv');a=p.parse_args()
 manifest=json.loads((ROOT/a.manifest).read_text())
 train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
 assert test.id.equals(sample.id)
 for f,h in manifest['data_sha256'].items():
  assert hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest()==h, f'Data changed: {f}'
 pred=np.zeros(len(test));total=0
 for component in manifest['components']:
  folder=ROOT/component['directory'];record=json.loads((folder/'result.json').read_text());cfg=record['config']
  assert record['status']=='complete' and record['split']=='full','Refuse partial or holdout models for a submission'
  assert record['best_iterations']==record['fixed_trees']==component['trees']
  assert component['weight']>0
  _,T,_=features(train,test,cfg['kind'],cfg.get('feature_mode'))
  if cfg['kind']=='catboost':
   mp=folder/'model.cbm';assert hashlib.sha256(mp.read_bytes()).hexdigest()==component['model_sha256']
   model=CatBoostClassifier();model.load_model(str(mp));assert model.tree_count_==component['trees']
   prob=model.predict_proba(T,thread_count=6)[:,1]
  else:
   import lightgbm as lgb
   mp=folder/'model.txt';assert hashlib.sha256(mp.read_bytes()).hexdigest()==component['model_sha256']
   model=lgb.Booster(model_file=str(mp));assert model.num_trees()==component['trees']
   prob=model.predict(T,num_threads=6)
  assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
  pred+=component['weight']*prob;total+=component['weight']
 assert abs(total-1)<1e-12
 sample['satisfaction']=pred;out=ROOT/a.output;sample.to_csv(out,index=False)
 back=pd.read_csv(out);assert back.id.equals(test.id)
 assert np.allclose(back.satisfaction,pred,rtol=0,atol=1e-15)
 print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'rows':len(back),'min':float(pred.min()),'max':float(pred.max())},indent=2))
if __name__=='__main__':main()

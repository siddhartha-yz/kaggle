"""Read-only audit of transferred official data, splits, and saved predictions."""
from pathlib import Path
import hashlib, json, platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
from iterate import features

root=Path(__file__).resolve().parent
meta=json.loads((root/'artifacts/experiment.json').read_text())
for name,digest in meta['data_sha256'].items():
    assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==digest,name
train=pd.read_csv(root/'data/train.csv'); test=pd.read_csv(root/'data/test.csv');sample=pd.read_csv(root/'data/sample_submission.csv')
assert train.shape==(699635,23) and test.shape==(299844,22)
assert list(sample.columns)==['id','satisfaction']
assert train.satisfaction.dtype==bool
assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
assert not set(train.id).intersection(test.id)
assert test.id.equals(sample.id)
expected=[c for c in train if c not in ['id','satisfaction']]
assert [c for c in test if c!='id']==expected and len(expected)==21
y=train.satisfaction.astype('int8')
fit,valid=train_test_split(np.arange(len(train)),test_size=.2,random_state=42,stratify=y)
split=np.load(root/'artifacts/split_seed42.npz')
assert np.array_equal(fit,split['train']) and np.array_equal(valid,split['valid'])
base=pd.read_csv(root/'artifacts/validation_predictions.csv')
assert np.array_equal(base.id,train.id.iloc[valid]) and np.array_equal(base.target,y.iloc[valid])
assert abs(roc_auc_score(base.target,base.probability_true)-meta['validation_auc'])<1e-14
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'rows_train':len(train),'rows_test':len(test),'feature_count':len(expected),'data_sha256':meta['data_sha256'],'split_seed42_reproduced':True,'primary_baseline_auc':meta['validation_auc'],'python':platform.python_version(),'saved_predictions':{}}
for name in ['cb6cross','cb7','cb8','lgb31','lgb31fe','lgb63','lgb63allcats','lgb63cats']:
    directory=root/'iteration_artifacts'/f'{name}_primary_s42'
    record=json.loads((directory/'result.json').read_text())
    v=np.load(directory/'validation.npz')
    assert record['status']=='complete' and np.array_equal(v['idx'],valid)
    assert np.array_equal(v['id'],base.id) and np.array_equal(v['target'],base.target)
    assert np.isfinite(v['pred']).all() and ((v['pred']>=0)&(v['pred']<=1)).all()
    auc=float(roc_auc_score(v['target'],v['pred']))
    assert abs(auc-record['validation_auc'])<1e-14
    report['saved_predictions'][name]={'auc':auc,'trees':record['best_iterations']}
# Cross-host CPU inference checks, no fitting.
ix=np.linspace(0,len(valid)-1,1003,dtype=int)
for name in ['cb8','lgb63cats']:
    directory=root/'iteration_artifacts'/f'{name}_primary_s42'
    record=json.loads((directory/'result.json').read_text()); cfg=record['config']
    X,_,_=features(train,test,cfg['kind'],cfg.get('feature_mode'))
    if name=='cb8':
        model=CatBoostClassifier();model.load_model(str(directory/'model.cbm'))
        pred=model.predict_proba(X.iloc[valid[ix]],thread_count=6)[:,1]
    else:
        model=lgb.Booster(model_file=str(directory/'model.txt'))
        pred=model.predict(X.iloc[valid[ix]],num_threads=6)
    stored=np.load(directory/'validation.npz')['pred'][ix]
    assert np.allclose(pred,stored,rtol=0,atol=1e-12),name
    report['saved_predictions'][name]['max_cross_host_inference_difference']=float(np.max(np.abs(pred-stored)))
p1=np.load(root/'iteration_artifacts/cb8_primary_s42/validation.npz')['pred']
p2=np.load(root/'iteration_artifacts/lgb63cats_primary_s42/validation.npz')['pred']
report['primary_frozen_blend_auc']=float(roc_auc_score(y.iloc[valid],.5*p1+.5*p2))
assert abs(report['primary_frozen_blend_auc']-.9581313490780615)<1e-14
(root/'local_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

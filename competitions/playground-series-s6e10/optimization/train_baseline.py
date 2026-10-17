"""Official-data-only CPU CatBoost baseline for Kaggle S6E10.

Run from this directory: .venv/bin/python -u train_baseline.py
The only source is the ZIP downloaded using Kaggle's logged-in download UI.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import time

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
SEED = 42

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

started = time.time()
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
sample = pd.read_csv(DATA / 'sample_submission.csv')
assert list(sample.columns) == ['id', 'satisfaction']
assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
assert not set(train.id).intersection(test.id)
assert test.id.equals(sample.id)
assert list(train.drop(columns='satisfaction').columns) == list(test.columns)
assert train.satisfaction.dtype == bool, 'Inspect labels before allowing a new mapping.'
assert set(train.satisfaction.unique()) == {False, True}
y = train.satisfaction.astype('int8')
# Raw CSV values are False/True; True is satisfaction. The sample constant
# matches this positive prevalence, providing a second mapping check.
assert np.isclose(sample.satisfaction.iloc[0], y.mean(), atol=1e-12)
X = train.drop(columns=['id', 'satisfaction']).copy()
Xtest = test.drop(columns='id').copy()
categorical = X.select_dtypes(include=['object', 'string']).columns.tolist()
for name in categorical:
    X[name] = X[name].fillna('__MISSING__').astype(str)
    Xtest[name] = Xtest[name].fillna('__MISSING__').astype(str)
assert 'id' not in X.columns and 'satisfaction' not in X.columns
assert X.columns.equals(Xtest.columns)
idx_train, idx_valid = train_test_split(np.arange(len(train)), test_size=0.20,
                                       random_state=SEED, stratify=y)
assert not set(idx_train).intersection(idx_valid)
np.savez_compressed(OUT / 'split_seed42.npz', train=idx_train, valid=idx_valid)
params = dict(iterations=1000, learning_rate=0.1, depth=6,
              loss_function='Logloss', eval_metric='AUC', random_seed=SEED,
              l2_leaf_reg=3.0, task_type='CPU', thread_count=6,
              allow_writing_files=False, verbose=100)
record = dict(experiment='catboost_cpu_baseline_v1', started_utc=datetime.now(timezone.utc).isoformat(),
              source_url='https://www.kaggle.com/competitions/playground-series-s6e10/data',
              external_data_used=False, target='satisfaction', positive_class='True',
              label_mapping={'False': 0, 'True': 1}, rows_train=len(train), rows_test=len(test),
              train_positive_rate=float(y.mean()), features=list(X.columns),
              categorical_features=categorical, missing_train=X.isna().sum().to_dict(),
              missing_test=Xtest.isna().sum().to_dict(),
              split=dict(method='stratified_random_holdout', seed=SEED, validation_fraction=0.2,
                         rows_fit=len(idx_train), rows_valid=len(idx_valid),
                         split_file_sha256=sha(OUT/'split_seed42.npz')),
              parameters=params, early_stopping_rounds=75,
              packages=dict(python=platform.python_version(), catboost=catboost.__version__,
                            pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
              data_sha256={f:sha(DATA/f) for f in ['train.csv','test.csv','sample_submission.csv']},
              stage='holdout_training')
write_json('experiment.json', record)
print('DATA_VALIDATED', json.dumps({k:record[k] for k in ['rows_train','rows_test','train_positive_rate','categorical_features','split']}), flush=True)
model = CatBoostClassifier(**params)
model.fit(X.iloc[idx_train], y.iloc[idx_train], cat_features=categorical,
          eval_set=(X.iloc[idx_valid], y.iloc[idx_valid]), early_stopping_rounds=75, use_best_model=True)
assert list(model.classes_) == [0,1]
pvalid = model.predict_proba(X.iloc[idx_valid])[:, 1]
auc = float(roc_auc_score(y.iloc[idx_valid], pvalid))
ll = float(log_loss(y.iloc[idx_valid], pvalid))
iterations = model.get_best_iteration() + 1
model.save_model(str(OUT / 'holdout_model.cbm'))
pd.DataFrame({'id':train.id.iloc[idx_valid].to_numpy(), 'target':y.iloc[idx_valid].to_numpy(),
              'probability_true':pvalid}).to_csv(OUT/'validation_predictions.csv', index=False)
record.update(validation_auc=auc, validation_logloss=ll, best_iterations=iterations,
              holdout_seconds=round(time.time()-started,2), stage='full_refit')
write_json('experiment.json', record)
print('VALIDATION_RESULT', json.dumps({k:record[k] for k in ['validation_auc','validation_logloss','best_iterations','holdout_seconds']}), flush=True)
full_params = dict(params, iterations=iterations)
final_model = CatBoostClassifier(**full_params)
final_model.fit(X, y, cat_features=categorical)
assert list(final_model.classes_) == [0, 1]
pred = final_model.predict_proba(Xtest)[:, 1]
assert pred.shape == (len(test),)
assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
assert np.std(pred) > 1e-6
submission = sample.copy()
submission['satisfaction'] = pred
assert submission.id.equals(test.id) and submission.id.equals(sample.id)
assert list(submission.columns) == list(sample.columns)
path = OUT/'submission_catboost_v1.csv'
submission.to_csv(path, index=False)
readback = pd.read_csv(path)
assert readback.shape == sample.shape and list(readback.columns) == list(sample.columns)
assert readback.id.equals(sample.id) and readback.id.is_unique
assert np.isfinite(readback.satisfaction).all() and readback.satisfaction.between(0,1).all()
assert np.allclose(readback.satisfaction, pred, rtol=0, atol=1e-15)
final_model.save_model(str(OUT/'full_model.cbm'))
pd.DataFrame({'feature':X.columns, 'importance':final_model.feature_importances_}).sort_values('importance', ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
record.update(stage='submission_ready', elapsed_seconds=round(time.time()-started,2),
              submission=dict(filename=path.name, rows=len(submission), columns=list(submission.columns),
                              sha256=sha(path), bytes=path.stat().st_size,
                              probability_min=float(pred.min()), probability_max=float(pred.max()),
                              probability_mean=float(pred.mean()), id_order_exact_match=True,
                              finite_probability_check=True, readback_check=True))
write_json('experiment.json', record)
print('SUBMISSION_READY', json.dumps(record['submission']), flush=True)

"""Synthetic pair direction/coverage, score-origin invariance and native CPU support."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,json
from scipy.special import expit
from catboost import CatBoostRanker,Pool
from threadpoolctl import threadpool_limits
from research_r67_pairs import pairs,intercept
rng=np.random.default_rng(20261067);X=rng.normal(size=(1024,5));y=(X[:,0]+.2*X[:,1]>0).astype('int8');graph,weights=pairs(y);assert np.array_equal(graph,pairs(y)[0]) and np.array_equal(weights,pairs(y)[1]);margin=rng.normal(size=len(y));b=intercept(margin,y);shifted=intercept(margin+7,y);assert abs(b-shifted-7)<1e-11 and np.max(np.abs(expit(margin+b)-expit(margin+7+shifted)))<1e-13
cfg=dict(iterations=20,depth=4,learning_rate=.1,loss_function='PairLogit',eval_metric='AUC',random_seed=42,task_type='CPU',thread_count=6,allow_writing_files=False,verbose=False)
with threadpool_limits(6):
 model=CatBoostRanker(**cfg);model.fit(Pool(X,label=y,group_id=np.zeros(len(y),'int64'),pairs=graph,pairs_weight=weights));score=model.predict(X)
assert score[y==1].mean()>score[y==0].mean()
print(json.dumps(dict(status='passed_synthetic_no_official_data',rows=len(y),weighted_pairs=int(weights.sum()),unique_pairs=len(graph),minimum_weighted_degree=float(np.bincount(graph.ravel(),weights=np.repeat(weights,2),minlength=len(y)).min()),mean_calibration_error=float(abs(expit(margin+b).mean()-y.mean())),score_origin_invariance_error=float(np.max(np.abs(expit(margin+b)-expit(margin+7+shifted)))),native_resolved_parameters=model.get_all_params(),CPU_threads=6,GPU_used=False)))

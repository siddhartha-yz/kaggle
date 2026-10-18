"""Complete-fold diagnostics only; no fitted weights, region selection or new models."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from scipy.special import logit
from research_r4_common import now,sha,dump
from research_r23_loader import pool

def main():
    df=pd.read_csv('data/train.csv');y=df.satisfaction.to_numpy('int8');b=np.load('research_artifacts/r23_conditional_lean_fivefold/oof.npz');t=np.load('research_artifacts/r21_probability_lgb_officialcb_fivefold/oof.npz');assert np.array_equal(b['id'],df.id) and np.array_equal(b['target'],y) and np.max(np.abs(pool(t['tree'],b['pred'])-b['candidate']))<1e-12
    pred={'tree':t['tree'],'NN74':b['pred'],'v11':b['candidate']}
    for route,path in [('R54','r54_tabpfn_v2_fivefold'),('R55','r55_randomized_lgb84_fivefold'),('R56','r56_cohortview_lean_fivefold')]:
        r=json.load(open(f'research_artifacts/{path}/result.json'));assert r['status']=='complete' and not r['eligible'];z=np.load(f'research_artifacts/{path}/oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y);pred[route]=z['candidate']
    rows=[]
    for key,index in [('ALL',np.arange(len(df))),*[(str(key),np.asarray(indices)) for key,indices in df.groupby(['Class','Type of Travel','Customer Type'],observed=True).indices.items()]]:
        if np.unique(y[index]).size<2:continue
        metrics={name:dict(auc=float(roc_auc_score(y[index],p[index])),logloss=float(log_loss(y[index],p[index],labels=[0,1]))) for name,p in pred.items()}
        delta=logit(np.clip(pred['NN74'][index],1e-6,1-1e-6))-logit(np.clip(pred['tree'][index],1e-6,1-1e-6));gradient=float(np.sum((pred['v11'][index]-y[index])*delta));foldgains=[]
        for k in range(5):
            q=index[b['fold'][index]==k];foldgains.append(float(roc_auc_score(y[q],pred['NN74'][q])-roc_auc_score(y[q],pred['tree'][q])) if np.unique(y[q]).size==2 else None)
        rows.append(dict(group=key,rows=len(index),positive_rate=float(y[index].mean()),metrics=metrics,NN_minus_tree_auc=metrics['NN74']['auc']-metrics['tree']['auc'],NN_minus_tree_byfold=foldgains,logloss_weight_gradient_at_equal_pool=gradient,mean_squared_logit_disagreement=float(np.mean(delta**2))))
    result=dict(status='completed_diagnostic_only',checked_utc=now(),all_fivefolds_complete_before_diagnostics=True,predefined_group_columns=['Class','Type of Travel','Customer Type'],no_new_predictions_models_or_blendweights_fitted=True,selection_bias_warning='Historically reused validation labels. Slice findings are hypothesis-generating, not heldout evidence; any next gate must fit only fresh inner heldout predictions inside each outer training split.',source_sha256={n:sha(n) for n in [__file__,'research_artifacts/r23_conditional_lean_fivefold/oof.npz','research_artifacts/r21_probability_lgb_officialcb_fivefold/oof.npz','research_r54_result.json','research_r55_result.json','research_r56_result.json']},groups=rows);dump('research_r57_error_analysis.json',result)
    for r in rows:print(r['group'],r['rows'],'NN-tree_AUC',round(r['NN_minus_tree_auc'],7),'folds',[round(d,6) if d is not None else None for d in r['NN_minus_tree_byfold']],'LL_weight_gradient',round(r['logloss_weight_gradient_at_equal_pool'],1),flush=True)
if __name__=='__main__':main()

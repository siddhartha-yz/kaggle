"""Source-only diagnostic: original SAT predictability from ratings vs nonrating context."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,threading,numpy as np,pandas as pd,lightgbm as lgb,joblib
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r60_source_data import data,RATINGS
from research_features import DistanceFeatures
def main():
    plan=json.loads((ROOT/'research_r62_support_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r62_support_resource_failure.json'),daemon=True);watcher.start()
    try:
        raw,y,train,test,cols,scope=data();assert not scope['official_SAT_values_parsed'];split=np.load(ROOT/'research_artifacts/original_teacher_v1/original_selection_split.npz');fi,vi=split['train'],split['valid'];assert not np.intersect1d(fi,vi).size and np.array_equal(np.sort(np.r_[fi,vi]),np.arange(129859));features=[c for c in cols if c not in RATINGS];assert len(features)==8
        with threadpool_limits(6):
            fullmap=DistanceFeatures('base').fit(raw.iloc[fi].reset_index(drop=True));fullmap.te_maps=[];full=lgb.Booster(model_file=str(ROOT/'research_artifacts/original_teacher_v1/selection_model.txt'));fullpred=full.predict(fullmap.transform_te(raw.iloc[vi].reset_index(drop=True)),num_threads=6);fullauc=float(roc_auc_score(y[vi],fullpred));old=json.loads((ROOT/'research_artifacts/original_teacher_v1/original_selection_result.json').read_text());assert abs(fullauc-old['auc'])<1e-12
            X=raw.iloc[fi][features].copy();V=raw.iloc[vi][features].copy();vocab={}
            for c in X.select_dtypes(exclude='number'):
                dtype=pd.CategoricalDtype(sorted(X[c].fillna('__MISSING__').astype(str).unique()));vocab[c]=dtype;X[c]=X[c].fillna('__MISSING__').astype(str).astype(dtype);V[c]=V[c].fillna('__MISSING__').astype(str).astype(dtype)
            m=lgb.LGBMClassifier(**plan['metadata_classifier_parameters']);m.fit(X,y[fi]);assert np.array_equal(m.classes_,[0,1]);pred=m.predict_proba(V)[:,1];out=ROOT/'research_artifacts/r62_source_support';out.mkdir(exist_ok=False);m.booster_.save_model(str(out/'metadata_model.txt'));joblib.dump(vocab,out/'metadata_vocab.joblib');replay=lgb.Booster(model_file=str(out/'metadata_model.txt')).predict(V,num_threads=6);assert np.max(np.abs(replay-pred))<1e-12
        r=dict(status='source_diagnostic_only_no_official_SAT_or_quality',checked_utc=now(),plan_sha256=sha(ROOT/'research_r62_support_plan.json'),source_scope=scope,source_holdout_rows=len(vi),full21_existing_model_AUC=fullauc,metadata8_new_model_AUC=float(roc_auc_score(y[vi],pred)),full21_logloss=float(log_loss(y[vi],fullpred)),metadata8_logloss=float(log_loss(y[vi],pred)),metadata_features=features,new_source_only_diagnostic_model_fits=1,no_new_full21_baseline_fit=True,metadata_model_sha256=sha(out/'metadata_model.txt'),metadata_vocab_sha256=sha(out/'metadata_vocab.joblib'),peak_RSS_bytes=peak[0],interpretation='Predictability contrast diagnoses the role of rating information in source SAT. It does not measure counterfactual support directly or establish utility on official SAT; any smaller-conditioning density route needs the original full5/3gates.');dump(ROOT/'research_r62_source_support.json',r);print('R62_SOURCE_SUPPORT_DIAGNOSTIC',r['full21_existing_model_AUC'],r['metadata8_new_model_AUC'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

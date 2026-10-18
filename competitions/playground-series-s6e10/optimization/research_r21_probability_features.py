"""Replay cached13/65 rating models on CPU, never fit or load satisfaction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,gc,resource
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,guard
from research_r19_aux_source import RATINGS
from research_r21_conditional import conditional_statistics

def main():
    variant=sys.argv[1];assert variant in ('sourcecb','officialcb')
    plan=json.loads((ROOT/'research_r21_plan.json').read_text())
    for kind in ('source_sha256','artifact_sha256'):
        for n,h in plan[kind].items():assert sha(ROOT/n)==h
    cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']]
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*cols]);test=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*cols])
    raw=pd.concat([train[cols],test[cols]],ignore_index=True);ids=np.r_[train.id.to_numpy(),test.id.to_numpy()]
    for c in raw.select_dtypes(exclude='number'):raw[c]=raw[c].fillna('__MISSING__').astype(str)
    src=ROOT/f'research_artifacts/r20_aux_{variant}';sr=json.loads((src/'result.json').read_text())
    assert sr['status']=='complete' and not sr['satisfaction_column_loaded'] and not sr['satisfaction_used_as_target_or_input']
    old=[]
    for name,frame in [('train',train),('test',test)]:
        z=np.load(src/f'{name}_features.npz');assert np.array_equal(z['id'],frame.id) and sha(src/f'{name}_features.npz')==sr[name+'_features_sha256'];old.append(z['values'])
    previous=np.concatenate(old);folds=np.load(src/'auxiliary_folds.npz');assert np.array_equal(folds['id'],ids)
    out=ROOT/f'research_artifacts/r21_conditional_{variant}';out.mkdir(exist_ok=True)
    allsurprisal=np.full((len(raw),13),np.nan,'float32');records=[]
    for row in sr['models']:
        guard();tick=time.monotonic();j,k=row['rating_index'],row['auxiliary_fold'];c=RATINGS[j];features=[x for x in cols if x!=c]
        vi=np.arange(len(raw)) if variant=='sourcecb' else np.flatnonzero(folds['fold']==k)
        run=src/f'rating{j}_fold{k}';assert row['model_sha256']==sha(run/'model.cbm') and row['feature_columns']==features and row['target_excluded_from_inputs']
        dump(ROOT/'research_r21_state.json',dict(status='conditional_probability_features',updated_utc=now(),variant=variant,rating_index=j,auxiliary_fold=k,pid=os.getpid(),active_work=True,active_training=False,device='cpu',cpu_threads=6))
        cache=out/f'rating{j}_fold{k}.npz'
        with threadpool_limits(6):
            model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'))
            assert model.tree_count_==300 and model.feature_names_==features and np.array_equal(model.classes_,row['classes'])
            probability=model.predict_proba(raw.iloc[vi][features],thread_count=6)
            expected,surprisal=conditional_statistics(probability,model.classes_,raw[c].to_numpy()[vi])
        error=float(np.max(np.abs(expected-previous[vi,j])));assert error<1e-6
        allsurprisal[vi,j]=surprisal;np.savez_compressed(cache,id=ids[vi],idx=vi,values=surprisal)
        r=dict(rating_index=j,auxiliary_fold=k,rows=len(vi),source_model_sha256=row['model_sha256'],prediction_sha256=sha(cache),
            expected_rating_replay_error=error,seconds=time.monotonic()-tick,observed_values_outside_source_classes=int((~np.isin(raw[c].to_numpy()[vi],model.classes_)).sum()))
        records.append(r);print('FEATURES',variant,j,k,r['seconds'],flush=True);del model,probability,expected,surprisal;gc.collect()
    assert np.isfinite(allsurprisal).all() and ((allsurprisal>=0)&(allsurprisal<=-np.log(1e-6)+1e-6)).all()
    values=np.column_stack([allsurprisal,allsurprisal.mean(axis=1,dtype='float64').astype('float32')])
    r=dict(status='complete',completed_utc=now(),variant=variant,plan_sha256=sha(ROOT/'research_r21_plan.json'),source_manifest_sha256=sha(src/'result.json'),
        source_models_used=len(records),records=records,no_new_models_fitted=True,satisfaction_loaded=False,transductive=(variant=='officialcb'),
        feature_columns=[f'rating_surprisal_{j}' for j in range(13)]+['mean_rating_surprisal'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for name,lo,hi,df in [('train',0,len(train),train),('test',len(train),len(raw),test)]:
        np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),values=values[lo:hi],columns=np.asarray(r['feature_columns']))
        r[name+'_features_sha256']=sha(out/f'{name}_features.npz')
    dump(out/'result.json',r)
if __name__=='__main__':main()

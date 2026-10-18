"""Recompute ranks without production formula; replay 65 cached posterior/class maps."""
import os,gc,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pandas as pd
from scipy.special import ndtri
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,sha
from research_r47_features import RATINGS

def verify_ranks():
    source=ROOT/'research_artifacts/r20_aux_officialcb';sr=json.loads((source/'result.json').read_text());posterior=ROOT/'research_artifacts/r25_posterior_officialcb';pr=json.loads((posterior/'result.json').read_text());ranks=ROOT/'research_artifacts/r37_ordinal_ranks';rr=json.loads((ranks/'result.json').read_text())
    assert not sr['satisfaction_column_loaded'] and not sr['satisfaction_used_as_target_or_input'] and len(sr['models'])==65
    assert pr['status']=='complete' and not pr['satisfaction_loaded'] and pr['no_new_models_fitted'] and pr['source_models_used']==65 and pr['source_manifest_sha256']==sha(source/'result.json')
    assert rr['posterior_manifest_sha256']==sha(posterior/'result.json') and not rr['satisfaction_column_loaded'] and rr['new_auxiliary_models_fitted']==0
    columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']];frames={name:pd.read_csv(ROOT/f'data/{name}.csv',usecols=['id',*columns]) for name in ['train','test']};post=[];rank_errors={}
    for name,frame in frames.items():
        z=np.load(posterior/f'{name}_features.npz');saved=np.load(ranks/f'{name}_features.npz');assert sha(posterior/f'{name}_features.npz')==pr[f'{name}_features_sha256'] and sha(ranks/f'{name}_features.npz')==rr[f'{name}_features_sha256'] and np.array_equal(z['id'],frame.id) and np.array_equal(saved['id'],frame.id)
        assert list(z['columns'])==[f'rating_posterior_{j}_class{k}' for j in range(13) for k in range(6)] and list(saved['columns'])==[f'conditional_positive_rank_{j}' for j in range(13)]
        probabilities=z['values'];errors=[]
        for j,c in enumerate(RATINGS):
            p=probabilities[:,j*6:(j+1)*6].astype('float64');observed=frame[c].to_numpy();mass=np.zeros(len(frame));less=np.zeros(len(frame));equal=np.zeros(len(frame))
            assert np.max(np.abs(p.sum(1)-1))<2e-6 and np.isfinite(p).all() and (p>=0).all()
            for k in range(1,6):
                mass+=p[:,k];less+=np.where(observed>k,p[:,k],0);equal+=np.where(observed==k,p[:,k],0)
            u=np.full(len(frame),.5);valid=(observed>0)&(mass>1e-12);u[valid]=(less[valid]+.5*equal[valid])/mass[valid];expected=ndtri(np.clip(u,1e-6,1-1e-6)).astype('float32');error=float(np.max(np.abs(expected-saved['values'][:,j])));assert error<1e-6;errors.append(error)
        rank_errors[name]=errors;post.append(probabilities);del z,saved,p,mass,less,equal;gc.collect()
    raw=pd.concat([frames['train'][columns],frames['test'][columns]],ignore_index=True);ids=np.r_[frames['train'].id,frames['test'].id];probabilities=np.concatenate(post);del post;gc.collect()
    for c in raw.select_dtypes(exclude='number'):raw[c]=raw[c].fillna('__MISSING__').astype(str)
    fold=np.full(len(raw),-1,'int8')
    for k,(_,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(raw)):fold[vi]=k
    sf=np.load(source/'auxiliary_folds.npz');assert np.array_equal(sf['id'],ids) and np.array_equal(sf['fold'],fold)
    records=[]
    for row in sr['models']:
        j,k=row['rating_index'],row['auxiliary_fold'];fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);modelpath=source/f'rating{j}_fold{k}/model.cbm';assert row['model_sha256']==sha(modelpath) and row['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest() and row['valid_index_sha256']==hashlib.sha256(vi.tobytes()).hexdigest() and not np.intersect1d(fi,vi).size
        linked=[r for r in pr['records'] if r['rating_index']==j and r['auxiliary_fold']==k];assert len(linked)==1 and linked[0]['source_model_sha256']==row['model_sha256']
        model=CatBoostClassifier(thread_count=6);model.load_model(str(modelpath));features=[c for c in columns if c!=RATINGS[j]];assert model.feature_names_==features and model.tree_count_==300 and row['feature_columns']==features and 'satisfaction' not in features and 'id' not in features
        query=np.random.default_rng(20261047+j*5+k).choice(vi,512,replace=False);classes=model.classes_.astype(int);assert np.array_equal(classes,np.sort(raw[RATINGS[j]].iloc[fi].unique()))
        with threadpool_limits(6):actual=model.predict_proba(raw.iloc[query][features],thread_count=6,task_type='CPU')
        canonical=np.zeros((len(query),6));canonical[:,classes]=actual;error=float(np.max(np.abs(canonical-probabilities[query,j*6:(j+1)*6])));assert error<1e-7;records.append(dict(rating_index=j,auxiliary_fold=k,canonical_probability_CPU_replay_error=error));del model;gc.collect()
    return dict(status='passed',rows=999479,official_SAT_values_parsed=False,positive_rating_order='1,2,3,4,5',state0='neutralrank0; separateordinalstate, not a missing-data claim',rank_max_errors=rank_errors,CPU_auxiliary_model_replays=records,posterior_manifest_sha256=sha(posterior/'result.json'),rank_manifest_sha256=sha(ranks/'result.json'))

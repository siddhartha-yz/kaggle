"""Rating distributions conditioned on eight nonrating covariates only; no SAT reads."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r46');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import gc,json,sys,time,threading,resource,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from research_r19_aux_source import RATINGS
from research_r4_common import now,dump,sha
from research_r23_guard import watch

def data():
    columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']]
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*columns]);test=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*columns])
    raw=pd.concat([train[columns],test[columns]],ignore_index=True);ids=np.r_[train.id.to_numpy(),test.id.to_numpy()]
    features=[c for c in columns if c not in RATINGS];assert len(features)==8 and len(columns)==21 and len(RATINGS)==13 and 'satisfaction' not in raw
    assert features==['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes','Gender','Customer Type','Type of Travel','Class']
    X=raw[features].copy();cats=X.select_dtypes(exclude='number').columns.tolist()
    for c in cats:X[c]=X[c].fillna('__MISSING__').astype(str)
    fold=np.full(len(raw),-1,'int8')
    for k,(_,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(raw)):fold[vi]=k
    assert raw.shape==(999479,21) and len(train)==699635 and len(test)==299844
    return ids,raw,X,cats,fold,len(train)

def statistics(probability,classes,observed):
    classes=np.asarray(classes,dtype=int);observed=np.asarray(observed,dtype=int)
    assert np.array_equal(classes,np.unique(classes)) and set(classes)<=set(range(6)) and probability.shape==(len(observed),len(classes))
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all() and np.max(np.abs(probability.sum(1)-1))<1e-6
    ev=(probability@classes).astype('float32')
    positions=np.searchsorted(classes,observed);assert (positions<len(classes)).all() and np.array_equal(classes[positions],observed)
    s=(-np.log(np.clip(probability[np.arange(len(observed)),positions],1e-6,1))).astype('float32')
    return ev,s

def main(probe=False):
    os.environ['CUDA_VISIBLE_DEVICES']='0' # imported historical RATINGS module sets it empty
    plan_name='research_r46_resource_plan.json' if probe else 'research_r46_plan.json'
    plan=json.loads((ROOT/plan_name).read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    if not probe:
        proof=json.loads((ROOT/'research_r46_probe.json').read_text());assert proof['status']=='passed_no_SAT_read_or_quality_score' and proof['peak_gpu_process_bytes']<=1073741824 and proof['parameters']==plan['auxiliary_parameters']
    ids,raw,X,cats,fold,ntrain=data();out=ROOT/'research_artifacts'/('r46_probe' if probe else 'r46_cohort_auxiliary');out.mkdir(exist_ok=True)
    np.savez_compressed(out/'auxiliary_folds.npz',id=ids,fold=fold)
    rec=dict(status='training',started_utc=now(),plan_sha256=sha(ROOT/plan_name),runner_sha256=sha(__file__),satisfaction_values_parsed=False,satisfaction_used_as_input_or_target=False,transductive_known_covariates=True,all_thirteen_ratings_excluded_from_inputs=True,feature_columns=list(X),categorical_columns=cats,parameters=plan['auxiliary_parameters'],auxiliary_fold_sha256=sha(out/'auxiliary_folds.npz'),models=[])
    evall=np.full((len(raw),13),np.nan,'float32');sall=evall.copy();peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        for j,rating in enumerate(RATINGS):
            for k in range(5):
                if probe and (j!=0 or k!=4):continue
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'rating{j}_fold{k}';run.mkdir(exist_ok=True)
                record=run/'result.json';cache=run/'predictions.npz';modelpath=run/'model.cbm'
                if record.exists() and json.loads(record.read_text()).get('status')=='complete':
                    row=json.loads(record.read_text());assert row['plan_sha256']==sha(ROOT/plan_name) and row['runner_sha256']==sha(__file__) and row['model_sha256']==sha(modelpath) and row['prediction_sha256']==sha(cache)
                    z=np.load(cache);assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],ids[vi]);evall[vi,j]=z['ev'];sall[vi,j]=z['surprisal'];rec['models'].append(row);continue
                tick=time.monotonic();dump(ROOT/'research_r46_state.json',dict(status='auxiliary_training',updated_utc=now(),rating_index=j,auxiliary_fold=k,pid=os.getpid(),active_training=True,probe=probe))
                assert not set(RATINGS)&set(X) and not np.intersect1d(fi,vi).size
                with threadpool_limits(6):
                    model=CatBoostClassifier(**plan['auxiliary_parameters']);model.fit(X.iloc[fi],raw[rating].iloc[fi],cat_features=cats)
                    assert model.get_all_params()['task_type']=='GPU' and model.tree_count_==300
                    p=model.predict_proba(X.iloc[vi],thread_count=6);ev,s=statistics(p,model.classes_,raw[rating].iloc[vi])
                    model.save_model(str(modelpath));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(modelpath));replay=loaded.predict_proba(X.iloc[vi[:2048]],thread_count=6);error=float(np.max(np.abs(p[:2048]-replay)));assert error<1e-12
                evall[vi,j]=ev;sall[vi,j]=s;np.savez_compressed(cache,id=ids[vi],idx=vi,ev=ev,surprisal=s,classes=model.classes_,probability=p.astype('float32'))
                row=dict(status='complete',rating=rating,rating_index=j,auxiliary_fold=k,train_rows=len(fi),valid_rows=len(vi),feature_columns=list(X),classes=model.classes_.tolist(),parameters=plan['auxiliary_parameters'],resolved_parameters=model.get_all_params(),train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),valid_index_sha256=hashlib.sha256(vi.tobytes()).hexdigest(),plan_sha256=sha(ROOT/plan_name),runner_sha256=sha(__file__),model_sha256=sha(modelpath),prediction_sha256=sha(cache),CPU_reload_error=error,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick)
                assert row['peak_rss_bytes']<=8589934592;dump(record,row);rec['models'].append(row);dump(out/'result.json',rec);print('AUX_DONE',j,k,row['seconds'],peak[0],flush=True);del model,loaded,p,replay;gc.collect()
        assert 0<peak[0]<=1073741824
        if probe:
            rec.update(status='passed_no_SAT_read_or_quality_score',completed_utc=now(),features=8,peak_gpu_process_bytes=peak[0],official_SAT_values_parsed=False,quality_scores_computed=False);dump(ROOT/'research_r46_probe.json',rec);return
        assert len(rec['models'])==65 and np.isfinite(evall).all() and np.isfinite(sall).all()
        values=np.column_stack([evall,sall,sall.mean(axis=1)]).astype('float32');columns=[f'cohort_rating_ev_{j}' for j in range(13)]+[f'cohort_rating_surprisal_{j}' for j in range(13)]+['cohort_rating_surprisal_mean']
        rec['features']={}
        for name,lo,hi in [('train',0,ntrain),('test',ntrain,len(raw))]:
            np.savez_compressed(out/f'{name}_features.npz',id=ids[lo:hi],values=values[lo:hi],columns=np.asarray(columns));rec['features'][name]=dict(rows=hi-lo,columns=columns,sha256=sha(out/f'{name}_features.npz'))
        rec.update(status='complete',completed_utc=now(),peak_gpu_process_bytes=peak[0]);dump(out/'result.json',rec)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main('--probe' in sys.argv)

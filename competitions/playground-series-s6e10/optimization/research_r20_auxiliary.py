"""GPU auxiliary rating reconstruction: source-only control vs official-feature crossfit.
Ideas: goodpjw2008 auxiliary-task features; Koumei Maki feature-ladder comparison.
No satisfaction column is loaded by this helper.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r20');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,sys,time,gc,threading,subprocess,resource,hashlib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from research_r19_aux_source import RATINGS
from research_r4_common import now,dump,sha,guard
CEILING=1073741824

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    variant=sys.argv[1];assert variant in ('sourcecb','officialcb')
    plan=json.loads((ROOT/'research_r20_plan.json').read_text())
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    for n,h in plan['artifact_sha256'].items():assert sha(ROOT/n)==h
    cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']]
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*cols]);test=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*cols])
    ids=np.r_[train.id.to_numpy(),test.id.to_numpy()];raw=pd.concat([train[cols],test[cols]],ignore_index=True)
    assert raw.shape==(999479,21) and 'satisfaction' not in raw
    source=pd.read_csv(ROOT/'research_sources/original_cc0/data.csv',usecols=cols)
    keep=np.load(ROOT/'research_sources/original_safe_mask.npy');assert keep.sum()==129859 and (~keep).sum()==21
    source=source.loc[keep].reset_index(drop=True)
    for frame in [source,raw]:
        for c in frame.select_dtypes(exclude='number'):frame[c]=frame[c].fillna('__MISSING__').astype(str)
    out=ROOT/f'research_artifacts/r20_aux_{variant}';out.mkdir(exist_ok=True)
    if (out/'result.json').exists() and json.loads((out/'result.json').read_text()).get('status')=='complete':return
    auxiliary_fold=np.full(len(raw),-1,'int8')
    for k,(_,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(raw)):auxiliary_fold[vi]=k
    np.savez_compressed(out/'auxiliary_folds.npz',id=ids,fold=auxiliary_fold)
    rec=dict(status='training',started_utc=now(),variant=variant,plan_sha256=sha(ROOT/'research_r20_plan.json'),
        runner_sha256=sha(__file__),satisfaction_column_loaded=False,satisfaction_used_as_target_or_input=False,
        official_covariates_used=variant=='officialcb',transductive=variant=='officialcb',source_safe_rows=129859,
        auxiliary_fold_sha256=sha(out/'auxiliary_folds.npz'),parameters=plan['auxiliary_parameters'],models=[])
    dump(out/'result.json',rec);allpred=np.full((len(raw),13),np.nan,'float32')
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(x.strip()) for x in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*2**20)
                    if peak[0]>CEILING:dump(out/'resource_failure.json',dict(peak_gpu_process_bytes=peak[0],ceiling=CEILING));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        for j,c in enumerate(RATINGS):
            base=source if variant=='sourcecb' else raw
            target=base[c].to_numpy('int8');features=[x for x in cols if x!=c];assert len(features)==20 and c not in features
            splits=[(np.arange(len(source)),np.arange(len(raw)))] if variant=='sourcecb' else [(np.flatnonzero(auxiliary_fold!=k),np.flatnonzero(auxiliary_fold==k)) for k in range(5)]
            for k,(fi,vi) in enumerate(splits):
                run=out/f'rating{j}_fold{k}';run.mkdir(exist_ok=True);record=run/'result.json';cache=run/'predictions.npz';modelpath=run/'model.cbm'
                if record.exists() and json.loads(record.read_text()).get('status')=='complete':
                    row=json.loads(record.read_text());assert row['model_sha256']==sha(modelpath) and row['prediction_sha256']==sha(cache)
                    assert row['variant']==variant and row['rating']==c and row['feature_columns']==features and row['parameters']==plan['auxiliary_parameters']
                    assert row['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest() and row['valid_index_sha256']==hashlib.sha256(vi.tobytes()).hexdigest()
                    p=np.load(cache);assert np.array_equal(p['idx'],vi) and np.array_equal(p['id'],ids[vi]);allpred[vi,j]=p['pred'];rec['models'].append(row);continue
                guard();tick=time.monotonic()
                if variant=='officialcb':assert not np.intersect1d(fi,vi).size
                dump(ROOT/'research_r20_state.json',dict(status='auxiliary_training',updated_utc=now(),variant=variant,rating=c,rating_index=j,auxiliary_fold=k,pid=os.getpid(),active_training=True,peak_gpu_process_bytes=peak[0]))
                X=base.iloc[fi][features];V=raw.iloc[vi][features];cats=X.select_dtypes(exclude='number').columns.tolist()
                with threadpool_limits(6):
                    model=CatBoostClassifier(**plan['auxiliary_parameters']);model.fit(X,target[fi],cat_features=cats)
                    assert model.tree_count_==300 and model.get_all_params()['task_type']=='GPU'
                    classes=model.classes_.astype('float32');assert np.array_equal(classes,np.sort(np.unique(target[fi])))
                    probability=model.predict_proba(V,thread_count=6);assert probability.shape==(len(vi),len(classes)) and np.isfinite(probability).all()
                    prediction=(probability@classes).astype('float32');assert ((prediction>=0)&(prediction<=5.000001)).all()
                    model.save_model(str(modelpath));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(modelpath))
                    again=loaded.predict_proba(V.iloc[:2048],thread_count=6)
                    error=float(np.max(np.abs(probability[:2048]-again)));assert error<1e-12
                np.savez_compressed(cache,id=ids[vi],idx=vi,pred=prediction);allpred[vi,j]=prediction
                row=dict(status='complete',variant=variant,rating_index=j,rating=c,auxiliary_fold=k,train_rows=len(fi),valid_rows=len(vi),
                    feature_columns=features,categorical_columns=cats,classes=classes.tolist(),target_excluded_from_inputs=True,
                    train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),valid_index_sha256=hashlib.sha256(vi.tobytes()).hexdigest(),
                    parameters=plan['auxiliary_parameters'],resolved_parameters=model.get_all_params(),model_sha256=sha(modelpath),prediction_sha256=sha(cache),
                    model_reload_error=error,seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                dump(record,row);rec['models'].append(row);dump(out/'result.json',rec)
                print('AUX_DONE',variant,j,k,row['seconds'],peak[0],flush=True)
                del model,loaded,X,V,probability,again;gc.collect()
        assert np.isfinite(allpred).all()
        for name,lo,hi,df in [('train',0,len(train),train),('test',len(train),len(raw),test)]:
            np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),values=allpred[lo:hi],columns=np.asarray([f'aux_rating_ev_{j}' for j in range(13)]))
            rec[name+'_features_sha256']=sha(out/f'{name}_features.npz')
        rec.update(status='complete',completed_utc=now(),peak_gpu_process_bytes=peak[0]);dump(out/'result.json',rec)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

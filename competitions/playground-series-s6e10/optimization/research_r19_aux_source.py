"""Original-only auxiliary rating reconstruction; no satisfaction target is fitted."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,gc,joblib,resource
import numpy as np,pandas as pd,xgboost as xgb
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,guard
from research_original import safe_original
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
def encode(raw,vocab):
    assert 'satisfaction' not in raw and 'id' not in raw
    out=raw.copy()
    for c,categories in vocab.items():out[c]=pd.Categorical(raw[c],categories=categories).codes
    return out.astype('float32')
def main():
    plan=json.loads((ROOT/'research_r19_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest
    out=ROOT/'research_artifacts/r19_original_aux';out.mkdir(exist_ok=True)
    if (out/'result.json').exists() and json.loads((out/'result.json').read_text()).get('status')=='complete':return
    source,train,test,cols,provenance=safe_original();raw=source[cols]
    assert len(raw)==129859 and all(c in raw for c in RATINGS)
    vocab={c:sorted(raw[c].dropna().unique().tolist()) for c in raw.select_dtypes(exclude='number')}
    joblib.dump(vocab,out/'vocab.joblib')
    X=encode(raw,vocab);A=encode(train[cols],vocab);B=encode(test[cols],vocab)
    predictions={'train':np.full((len(train),13),np.nan,'float32'),'test':np.full((len(test),13),np.nan,'float32')}
    rec=dict(status='training',started_utc=now(),plan_sha256=sha(ROOT/'research_r19_plan.json'),runner_sha256=sha(__file__),
        source_provenance=provenance,source_rows=len(raw),feature_columns=cols,vocab_sha256=sha(out/'vocab.joblib'),
        satisfaction_used_as_auxiliary_target_or_input=False,models=[],cpu_threads=6,device='cpu')
    dump(out/'result.json',rec)
    for j,c in enumerate(RATINGS):
        guard();tick=time.monotonic();features=[x for x in cols if x!=c];assert len(features)==20 and c not in features
        values=np.sort(raw[c].dropna().unique());assert len(values)>=2 and raw[c].notna().all()
        target=np.searchsorted(values,raw[c].to_numpy()).astype('int32')
        path=out/f'rating{j}.ubj';record=out/f'rating{j}.json';cache=out/f'rating{j}.npz'
        if record.exists() and json.loads(record.read_text()).get('status')=='complete':
            row=json.loads(record.read_text());assert row['model_sha256']==sha(path) and row['predictions_sha256']==sha(cache)
            saved=np.load(cache);assert np.array_equal(saved['train_id'],train.id) and np.array_equal(saved['test_id'],test.id)
            predictions['train'][:,j]=saved['train'];predictions['test'][:,j]=saved['test'];rec['models'].append(row);continue
        dump(ROOT/'research_r19_state.json',dict(status='source_rating_training',updated_utc=now(),rating=c,rating_index=j,pid=os.getpid(),active_training=True,device='cpu',cpu_threads=6))
        params=dict(plan['auxiliary_parameters'],num_class=len(values))
        with threadpool_limits(6):
            dm=xgb.QuantileDMatrix(X[features],label=target,nthread=6,max_bin=256)
            model=xgb.train(params,dm,num_boost_round=300);model.save_model(path)
            loaded=xgb.Booster();loaded.load_model(path);loaded.set_param({'nthread':6,'device':'cpu'})
            errors=[]
            for name,frame in [('train',A),('test',B)]:
                query=xgb.DMatrix(frame[features],nthread=6)
                prob=model.predict(query);assert prob.shape==(len(frame),len(values)) and np.isfinite(prob).all()
                assert np.max(np.abs(prob.sum(1)-1))<1e-6
                predictions[name][:,j]=prob@values.astype('float32')
                sample=xgb.DMatrix(frame[features].iloc[:2048],nthread=6)
                errors.append(float(np.max(np.abs(model.predict(sample)-loaded.predict(sample)))))
            assert max(errors)<1e-7
        np.savez_compressed(cache,train_id=train.id.to_numpy(),test_id=test.id.to_numpy(),train=predictions['train'][:,j],test=predictions['test'][:,j])
        row=dict(status='complete',rating=c,feature_columns=features,classes=values.tolist(),target_excluded_from_inputs=True,
            source_rows=len(raw),fixed_rounds=300,parameters=params,model_sha256=sha(path),predictions_sha256=sha(cache),
            reload_error=max(errors),seconds=time.monotonic()-tick)
        dump(record,row);rec['models'].append(row);dump(out/'result.json',rec);print('AUX_DONE',j,c,row['seconds'],flush=True)
        del dm,model,loaded,query,prob,sample;gc.collect()
    for name,df in [('train',train),('test',test)]:
        p=predictions[name];assert np.isfinite(p).all() and ((p>=0)&(p<=5.000001)).all()
        np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),values=p,columns=np.asarray([f'original_aux_ev_{j}' for j in range(13)]))
        rec[name+'_features_sha256']=sha(out/f'{name}_features.npz')
    rec.update(status='complete',completed_utc=now(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dump(out/'result.json',rec)
if __name__=='__main__':main()

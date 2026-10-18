"""Fixed GPU CatBoost views; replace only the existing CB10 tree-family share."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r16')
Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,sys,time,gc,threading,subprocess,signal,resource,hashlib
import numpy as np,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_r4_common import now,dump,sha,guard
from research_r5_auto_te import data
from research_external_ablation import external
from research_r16_cb_features import CatDistanceFeatures
from research_r15_loader import pool
CEILING=1073741824

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    variant=sys.argv[1];assert variant in ['numeric57','nominal58']
    stage='threefold' if '--threefold' in sys.argv else 'fivefold'
    plan=json.loads((ROOT/'research_r16_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest
    assert json.loads((ROOT/'research_artifacts/r16_cb_gpu_speed64_04/result.json').read_text())['status']=='passed_no_quality_score'
    guard();df,raw,y,fold,*_=data(stage)
    reference=np.load(ROOT/f'research_artifacts/r14_log_pool_{stage}/oof.npz')
    assert all(np.array_equal(reference[key],value) for key,value in [('id',df.id),('target',y),('fold',fold)])
    if stage=='fivefold':
        previous=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(previous['id'],df.id);old_cb=previous['pred']
    else:
        previous=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz');assert np.array_equal(previous['id'],df.id) and np.array_equal(previous['fold'],fold);old_cb=previous['components'][:,3]
    ext,_=external('both',df.id);out=ROOT/'research_artifacts'/f'r16_gpu_cb_{variant}_{stage}';out.mkdir(exist_ok=True)
    peak=[0];stop=threading.Event();predall=np.full(len(y),np.nan)
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:
                        dump(out/'resource_failure.json',dict(status='GPU_ceiling_exceeded',completed_utc=now(),peak_gpu_process_bytes=peak[0]));os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        for k in sorted(np.unique(fold)):
            k=int(k);fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            run=out/f'fold{k}';run.mkdir(exist_ok=True);record=run/'result.json'
            if record.exists() and json.loads(record.read_text()).get('status')=='complete':
                rec=json.loads(record.read_text());assert rec['model_sha256']==sha(run/'model.cbm') and rec['runner_sha256']==sha(__file__)
                z=np.load(run/'validation.npz');assert np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);predall[vi]=z['pred'];continue
            guard()
            def timeout(signum,frame):raise TimeoutError('R16perfit900seconds')
            signal.signal(signal.SIGALRM,timeout);signal.alarm(900);tick=time.monotonic()
            tr=CatDistanceFeatures(variant=='nominal58')
            X=tr.fit_transform(raw.iloc[fi].reset_index(drop=True),y[fi],ext.iloc[fi].reset_index(drop=True))
            V=tr.transform(raw.iloc[vi].reset_index(drop=True),ext.iloc[vi].reset_index(drop=True));joblib.dump(tr,run/'transformer.joblib')
            rec=dict(status='training',started_utc=now(),variant=variant,stage=stage,fold=k,train_rows=len(fi),valid_rows=len(vi),feature_width=X.shape[1],
                category_columns=tr.cat_columns,parameters=plan['parameters'],runner_sha256=sha(__file__),plan_sha256=sha(ROOT/'research_r16_plan.json'),
                train_index_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),outer_validation_used_in_fit=False)
            dump(record,rec);dump(ROOT/'research_r16_state.json',dict(rec,pid=os.getpid(),active_training=True));print('START',variant,stage,k,flush=True)
            try:
                with threadpool_limits(6):
                    model=CatBoostClassifier(**plan['parameters']);model.fit(X,y[fi],cat_features=tr.cat_columns)
                    assert model.tree_count_==813 and np.array_equal(model.classes_,[0,1])
                    resolved=model.get_all_params();assert resolved['task_type']=='GPU' and resolved['max_ctr_complexity']==1
                    pred=model.predict_proba(V,thread_count=6)[:,1];assert np.isfinite(pred).all()
                    model.save_model(str(run/'model.cbm'));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(run/'model.cbm'))
                    replay=loaded.predict_proba(V,thread_count=6)[:,1];error=float(np.max(np.abs(pred-replay)));assert error<1e-12
                tree=reference['tree'][vi]+.25*(pred-old_cb[vi]);candidate=pool(tree,reference['neural'][vi])
                assert ((tree>=0)&(tree<=1)).all() and np.isfinite(candidate).all()
                np.savez_compressed(run/'validation.npz',id=df.id.iloc[vi].to_numpy(),target=y[vi],idx=vi,pred=pred,candidate=candidate)
                predall[vi]=pred
                rec.update(status='complete',completed_utc=now(),seconds=time.monotonic()-tick,
                    model_sha256=sha(run/'model.cbm'),transformer_sha256=sha(run/'transformer.joblib'),prediction_sha256=sha(run/'validation.npz'),
                    model_reload_error=error,resolved_parameters=resolved,peak_gpu_process_bytes=peak[0],
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                assert rec['peak_rss_bytes']<8*1024**3
                dump(record,rec);print('DONE',variant,stage,k,rec['seconds'],flush=True)
                del model,loaded,X,V,tr;gc.collect()
            except BaseException as error:
                rec.update(status='failed',completed_utc=now(),error=repr(error));dump(record,rec);raise
            finally:signal.alarm(0)
        assert np.isfinite(predall).all();tree=reference['tree']+.25*(predall-old_cb);candidate=pool(tree,reference['neural'])
        np.savez_compressed(out/'oof.npz',id=df.id.to_numpy(),target=y,fold=fold,pred=predall,tree=tree,candidate=candidate,reference_v9=reference['candidate'])
        dump(out/'completed.json',dict(status='complete',completed_utc=now(),oof_sha256=sha(out/'oof.npz'),peak_gpu_process_bytes=peak[0]))
        dump(ROOT/'research_r16_state.json',dict(status='stage_complete',updated_utc=now(),variant=variant,stage=stage,active_training=False))
    finally:signal.alarm(0);stop.set();watcher.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump(ROOT/'research_r16_failure.json',dict(status='failed',completed_utc=now(),error=repr(error),active_training=False));raise

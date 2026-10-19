"""Gated selected-GPU-CB full fit, existing-family reconstruction and checked CSV."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r16')
Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,time,gc,threading,subprocess,signal,resource
import numpy as np,pandas as pd,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
from research_r4_common import now,dump,sha,guard
from research_r16_cb_features import CatDistanceFeatures
from research_external_ablation import external
from research_r15_loader import pool
CEILING=1073741824

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    finalplan=json.loads((ROOT/'research_r16_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256']==sha(ROOT/'research_r16_plan.json')
    for name,digest in finalplan['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=json.loads((ROOT/'research_r16_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in plan['artifact_sha256'].items():assert sha(ROOT/name)==digest
    audit=json.loads((ROOT/'research_r16_pre_inference_audit.json').read_text())
    assert audit['status']=='passed' and audit['verified_new_model_records']==8
    variant=audit['variant'];assert variant==json.loads((ROOT/'research_r16_selected.json').read_text())['variant']
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r16_gpu_cb_{variant}_{stage}/result.json').read_text())['eligible']
    for name,digest in [('test','ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'),('sample_submission','6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71')]:assert sha(ROOT/'data'/f'{name}.csv')==digest
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert train.satisfaction.dtype==bool and len(train)==699635 and len(test)==299844
    assert test.id.is_unique and np.array_equal(sample.id,test.id)
    raw=train.drop(columns=['id','satisfaction']);rawtest=test.drop(columns='id');assert list(raw)==list(rawtest)
    y=train.satisfaction.to_numpy('int8');ext,_=external('both',train.id)
    z=np.load(ROOT/'research_artifacts/original_teacher_v1/test_features.npz');assert np.array_equal(z['id'],test.id)
    ext_test=pd.DataFrame(z['values'],columns=z['columns'])
    out=ROOT/f'research_artifacts/r16_gpu_cb_{variant}_full';out.mkdir(exist_ok=True)
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:os._exit(86)
    def timeout(signum,frame):raise TimeoutError('FullGPUCB900seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();tick=time.monotonic()
    try:
        guard();tr=CatDistanceFeatures(variant=='nominal58');X=tr.fit_transform(raw,y,ext);V=tr.transform(rawtest,ext_test)
        assert X.shape[1]==(58 if tr.nominal else 57)
        with threadpool_limits(6):
            model=CatBoostClassifier(**plan['parameters']);model.fit(X,y,cat_features=tr.cat_columns)
            assert model.tree_count_==813 and np.array_equal(model.classes_,[0,1]) and model.get_all_params()['task_type']=='GPU'
            pred=model.predict_proba(V,thread_count=6)[:,1];model.save_model(str(out/'model.cbm'));joblib.dump(tr,out/'transformer.joblib')
            loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(out/'model.cbm'))
            error=float(np.max(np.abs(pred-loaded.predict_proba(V,thread_count=6)[:,1])));assert error<1e-12
        assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz');nn=np.load(ROOT/'research_artifacts/r4_lean_cv_final/test_components.npz')
        assert np.array_equal(core['id'],test.id) and np.array_equal(nn['id'],test.id)
        oldtree=.46875*core['lgb_original']+.09375*core['cb_original']+.1875*core['residual_prior05']+.25*core['cb_deep']
        assert np.max(np.abs(oldtree-nn['v5']))<1e-12
        newtree=.46875*core['lgb_original']+.09375*core['cb_original']+.1875*core['residual_prior05']+.25*pred
        assert np.max(np.abs(newtree-(oldtree+.25*(pred-core['cb_deep']))))<1e-12
        assert sha(ROOT/'submission_log_odds_pool_r14_v9.csv')=='55a1cc1805674a4d1419d1c9ff61b64ee35d61f0117782b6c886dd103f9f57c8'
        previous=pd.read_csv(ROOT/'submission_log_odds_pool_r14_v9.csv');assert np.array_equal(previous.id,test.id)
        assert np.max(np.abs(previous.satisfaction-pool(oldtree,nn['realmlp'])))<1e-14
        candidate=pool(newtree,nn['realmlp'])
        independent=expit(.5*logit(np.clip(newtree,1e-6,1-1e-6))+.5*logit(np.clip(nn['realmlp'],1e-6,1-1e-6)))
        assert np.max(np.abs(candidate-independent))<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),newCB=pred,newtree=newtree,oldtree=oldtree,oldNN=nn['realmlp'],candidate=candidate)
        csv=ROOT/'submission_gpu_catboost_r16_v10.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id)
        roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert roundtrip<1e-14
        result=dict(status='passed',audited_utc=now(),variant=variant,csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class=True,
            full_model_sha256=sha(out/'model.cbm'),full_transformer_sha256=sha(out/'transformer.joblib'),full_test_components_sha256=sha(out/'test_components.npz'),
            formula='tree_new=.46875LGB+.09375CB8+.1875residual+.25newGPUCB; equal-logodds poolwitholdleanCV40',
            model_reload_error=error,CSV_roundtrip_error=roundtrip,seconds=time.monotonic()-tick,
            peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            validation=audit['validation'],pre_inference_audit_sha256=sha(ROOT/'research_r16_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r16_plan.json'),
            formal_used=1,remaining=1,parent_only_submitter=True,actual_top100_verified=False,
            limitation='Selected after two prespecified views on historically reused labels; conditional bounds omit adaptive history. GPUtrainingmaybenondeterministic; actualKagglegainnotverified.')
        assert peak[0]<=CEILING and result['peak_rss_bytes']<8*1024**3
        dump(out/'result.json',result);dump(ROOT/'research_r16_delivery_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='validation'},indent=2),flush=True)
    finally:signal.alarm(0);stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

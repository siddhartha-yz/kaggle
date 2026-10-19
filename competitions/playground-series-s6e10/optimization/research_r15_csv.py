"""Audited CV160 log-opinion pooling with one cross-fitted customer prior; parent submits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_common import ROOT,now,dump,sha
import json,gc,time,threading,subprocess,resource
import numpy as np,pandas as pd,torch,joblib,cloudpickle
from threadpoolctl import threadpool_limits
from research_r8_probe import CAP,CEILING
from research_r15_loader import pool
from scipy.special import expit,logit

def main():
    finalplan=json.loads((ROOT/'research_r15_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256']==sha(ROOT/'research_r15_plan.json')
    for name,digest in finalplan['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=json.loads((ROOT/'research_r15_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    audit=json.loads((ROOT/'research_r15_pre_inference_audit.json').read_text())
    assert audit['status']=='passed' and audit['verified_model_records']==8
    assert audit['plan_sha256']==sha(ROOT/'research_r15_plan.json')
    for name,digest in audit['data_sha256'].items():assert sha(ROOT/'data'/f'{name}.csv')==digest
    for stage in ['fivefold','threefold']:
        assert json.loads((ROOT/f'research_artifacts/r15_segment_lean_{stage}/result.json').read_text())['eligible']
    test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id)
    raw=test.drop(columns='id');assert 'satisfaction' not in raw
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r15_segment_lean_cv_final';out.mkdir(exist_ok=True)
    source=ROOT/'research_artifacts/r15_segment_lean_fivefold'
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    total=np.zeros(len(test));receipts=[]
    try:
        for k in range(5):
            run=source/f'fold{k}';rec=json.loads((run/'result.json').read_text())
            assert rec['status']=='complete' and rec['model_sha256']==sha(run/'model.pkl')
            assert rec['transformer_sha256']==sha(run/'transformer.joblib')
            tr=joblib.load(run/'transformer.joblib');assert list(raw)==tr.columns;V=tr.transform(raw)
            cache=out/f'fold{k}.npz';receipt=out/f'fold{k}.json';tick=time.monotonic()
            if cache.exists():
                previous=json.loads(receipt.read_text());assert previous['model_sha256']==rec['model_sha256'] and previous['prediction_sha256']==sha(cache)
                z=np.load(cache);assert np.array_equal(z['id'],test.id);pred=z['pred']
            else:
                assert torch.cuda.mem_get_info()[0]>=1024**3
                with open(run/'model.pkl','rb') as handle:model=cloudpickle.load(handle)
                assert np.array_equal(model.classes_,[0,1]) and model.get_config()['n_ens']==32
                assert model.cv_alg_interface_.model.config['stop_epoch']==3
                model.to('cuda:0')
                with threadpool_limits(6):pred=model.predict_proba(V)[:,1]
                assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
                np.savez_compressed(cache,id=test.id.to_numpy(),pred=pred)
                del model;gc.collect();torch.cuda.empty_cache()
            row=dict(fold=k,native_members=32,fixed_epochs=3,
                model_sha256=rec['model_sha256'],prediction_sha256=sha(cache),seconds=time.monotonic()-tick)
            dump(receipt,row);receipts.append(row);total+=pred
            dump(out/'state.json',dict(status='predicting',updated_utc=now(),completed_folds=len(receipts),gpu_process_peak_bytes=peak[0]))
            print('PREDICTED',k,flush=True)
            del tr,V;gc.collect()
        neural=total/5
        core=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz')
        old=np.load(ROOT/'research_artifacts/r4_lean_cv_final/test_components.npz')
        assert np.array_equal(core['id'],test.id) and np.array_equal(old['id'],test.id)
        v5=(60*core['lgb_original']+12*core['cb_original']+24*core['residual_prior05']+32*core['cb_deep'])/128
        assert np.max(np.abs(v5-old['v5']))<1e-12
        assert sha(ROOT/'submission_log_odds_pool_r14_v9.csv')=='55a1cc1805674a4d1419d1c9ff61b64ee35d61f0117782b6c886dd103f9f57c8'
        oldcsv=pd.read_csv(ROOT/'submission_log_odds_pool_r14_v9.csv')
        assert np.array_equal(oldcsv.id,test.id) and np.max(np.abs(oldcsv.satisfaction-pool(v5,old['realmlp'])))<1e-14
        candidate=pool(v5,neural)
        independent_formula=expit(.5*logit(np.clip(v5,1e-6,1-1e-6))+.5*logit(np.clip(neural,1e-6,1-1e-6)))
        assert np.max(np.abs(candidate-independent_formula))<1e-12
        assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v5=v5,realmlp=neural,candidate=candidate)
        csv=ROOT/'submission_customer_prior_r15_v10.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id)
        error=float(np.max(np.abs(check.satisfaction-candidate)));assert error<1e-14
        # Reconstruct the final neural vector independently from all five caches.
        independent=sum(np.load(out/f'fold{k}.npz')['pred'] for k in range(5))/5
        assert np.max(np.abs(independent-neural))<1e-12
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,
            rows=len(check),columns=list(check),positive_class=True,native_final_members=160,outer_folds=5,
            fixed_epochs=3,model_records=receipts,
            family_logodds_weights={'v5_tree':.5,'newNN_CV160':.5},tree_probability_weights={'LGB':.46875,'CB8':.09375,'residual':.1875,'CB10':.25},
            CSV_roundtrip_error=error,gpu_process_peak_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            pre_inference_audit_sha256=sha(ROOT/'research_r15_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r15_plan.json'),
            validation=audit['validation'],formal_used=1,remaining=1,parent_only_submitter=True,actual_top100_verified=False,
            limitation='Qualified conditional offline evidence on historically reused labels; no leaderboard gain or Top100 guarantee.')
        dump(out/'result.json',result);dump(ROOT/'research_r15_delivery_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('model_records','validation')},indent=2),flush=True)
    finally:stop.set();watcher.join(timeout=5)

if __name__=='__main__':main()

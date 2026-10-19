"""Qualified audited frozen NN74 CV160 inference with bounded GPU, no uploads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_common import ROOT,now,dump,sha
from research_r61_loader import auxiliary,pool
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
import numpy as np,pandas as pd,torch,cloudpickle,joblib,json,time,gc,threading,resource
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit

def main():
    final=json.loads((ROOT/'research_r61_final_inference_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r61_plan.json')
    for n,h in {**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    plan=json.loads((ROOT/'research_r61_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r61_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_model_records']==8 and audit['verified_native_neural_members']==256 and audit['plan_sha256']==sha(ROOT/'research_r61_plan.json')
    for stage in ('fivefold','threefold'):assert json.loads((ROOT/f'research_artifacts/r61_sourceclass_lean_{stage}/result.json').read_text())['eligible']
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r61_sourceclass_nn_full';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and raw.shape[1]==21 and 'satisfaction' not in raw
        aux=auxiliary(test.id,test=True);total=np.zeros(len(test),'float64');records=[];tick=time.monotonic()
        for k in range(5):
            src=ROOT/f'research_artifacts/r61_sourceclass_lean_fivefold/fold{k}';rec=json.loads((src/'result.json').read_text());assert rec['status']=='complete' and rec['config']==plan['config']
            assert rec['model_sha256']==sha(src/'model.pkl') and rec['transformer_sha256']==sha(src/'transformer.joblib')
            cache=out/f'fold{k}.npz';tr=joblib.load(src/'transformer.joblib');V=tr.transform(raw,aux);assert V.shape==(299844,74);started=time.monotonic()
            if cache.exists():
                z=np.load(cache);assert np.array_equal(z['id'],test.id) and str(z['model_sha256'])==rec['model_sha256'] and str(z['transformer_sha256'])==rec['transformer_sha256'] and str(z['final_plan_sha256'])==sha(ROOT/'research_r61_final_inference_plan.json');pred=z['pred'];repeat=None
            else:
                torch.cuda.reset_peak_memory_stats();assert torch.cuda.mem_get_info()[0]>=1024**3
                with open(src/'model.pkl','rb') as handle:m=cloudpickle.load(handle)
                assert np.array_equal(m.classes_,[0,1]) and m.cv_alg_interface_.model.config['n_ens']==32 and m.cv_alg_interface_.model.progress.max_epochs==3
                m.to('cuda:0');assert all(p.device.type=='cuda' for p in m.cv_alg_interface_.model.parameters())
                with threadpool_limits(6):pred=m.predict_proba(V)[:,1];again=m.predict_proba(V.iloc[:2048])[:,1]
                repeat=float(np.max(np.abs(again-pred[:2048])));assert repeat<1e-7
                np.savez_compressed(cache,id=test.id.to_numpy(),pred=pred,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],final_plan_sha256=sha(ROOT/'research_r61_final_inference_plan.json'))
                del m;gc.collect();torch.cuda.empty_cache()
            assert pred.shape==(299844,) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();total+=pred
            records.append(dict(fold=k,native_members=32,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],prediction_sha256=sha(cache),repeat_error=repeat,seconds=time.monotonic()-started,peak_gpu_process_bytes=peak[0]));dump(out/'state.json',dict(status='predicting',completed_models=len(records),records=records,updated_utc=now()));print('PREDICTED',k,records[-1],flush=True);del tr,V;gc.collect()
        neural=total/5;old=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');assert np.array_equal(old['id'],test.id)
        official=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv');current=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(current['id'],test.id) and np.array_equal(official.id,test.id) and np.max(np.abs(pool(old['newtree'],current['newNN'])-official.satisfaction))<1e-14
        candidate=pool(old['newtree'],neural);equiv=expit(.5*logit(np.clip(old['newtree'],1e-6,1-1e-6))+.5*logit(np.clip(neural,1e-6,1-1e-6)));formula_error=float(np.max(np.abs(candidate-equiv)));assert formula_error<1e-12
        assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all() and np.std(candidate)>.01
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v10tree=old['newtree'],newNN=neural,oldNN=current['newNN'],candidate=candidate)
        csv=ROOT/'submission_source_class_neural_r61_v12.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id) and check.id.is_unique
        roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert roundtrip<1e-14 and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all()
        r=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class=True,models=records,native_test_members=160,formula='Equal logodds(currentv10fulltree,mean5savednative32NN74);clip1e-6 beforelogit;meanNNbeforelogpool',CSV_roundtrip_error=roundtrip,formula_error=formula_error,test_components_sha256=sha(out/'test_components.npz'),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,pre_inference_audit_sha256=sha(ROOT/'research_r61_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r61_plan.json'),final_plan_sha256=sha(ROOT/'research_r61_final_inference_plan.json'),validation=audit['validation'],new_formal_submissions=0,remaining_daily_quota=None,additional_qualified_submissions_authorized=True,parent_only_submitter=True,actual_top500_verified=False,limitation='ReusedSATvalidationlabels;conditionalboundsomitadaptivehistory. OOF32membersperrow,test160averagechangesinferencevariance. No leaderboardclaim.')
        assert 0<peak[0]<=CEILING;dump(out/'result.json',r);dump(ROOT/'research_r61_delivery_audit.json',r);print('CHECKED_CSV',csv.name,r['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as error:dump(ROOT/'research_r61_csv_failure.json',dict(error=repr(error),updated_utc=now()));raise

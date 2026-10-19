"""Audited saved-model inference: calibrate each native32 group, meanfive, fixedv11 pool."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_common import ROOT,now,dump,sha
from research_r23_loader import auxiliary,pool
from research_r68_calibration import predict_raw,calibrate
from research_r68_pure_loss import pure_ranknet
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import json,time,threading,gc,resource,cloudpickle,joblib,numpy as np,pandas as pd,torch

def main():
    final=json.loads((ROOT/'research_r68_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r68_plan.json').read_text());audit=json.loads((ROOT/'research_r68_pre_inference_audit.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r68_plan.json') and audit['status']=='passed' and audit['verified_model_records']==8 and audit['verified_native_neural_members']==256
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['input_sha256']}.items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r68_pure_ranknet_full';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and raw.shape[1]==21 and 'satisfaction' not in test;aux=auxiliary(test.id,test=True);total=np.zeros(len(test),'float64');records=[];new_predictions=0
        for k in range(5):
            src=ROOT/f'research_artifacts/r68_pure_ranknet_fivefold/fold{k}';rec=json.loads((src/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r68_plan.json')
            for n,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('calibration.json','calibration_sha256')]:assert sha(src/n)==rec[key]
            calibration=json.loads((src/'calibration.json').read_text());tr=joblib.load(src/'transformer.joblib');V=tr.transform(raw,aux);assert V.shape==(299844,74);cache=out/f'fold{k}.npz';receipt=out/f'fold{k}_receipt.json';started=time.monotonic()
            if receipt.exists():
                prior=json.loads(receipt.read_text());assert prior['status']=='complete' and prior['cache_sha256']==sha(cache) and prior['final_plan_sha256']==sha(ROOT/'research_r68_final_inference_plan.json');z=np.load(cache);assert np.array_equal(z['id'],test.id);raw_pred=z['raw_pred'];pred=z['pred'];assert np.array_equal(calibrate(raw_pred,calibration['bias']),pred);repeat=None
            else:
                assert torch.cuda.mem_get_info()[0]>=1024**3;torch.cuda.reset_peak_memory_stats()
                with open(src/'model.pkl','rb') as f:model=cloudpickle.load(f)
                assert np.array_equal(model.classes_,[0,1]) and model.cv_alg_interface_.model.criterion is pure_ranknet and model.cv_alg_interface_.model.config['n_ens']==32 and model.cv_alg_interface_.model.progress.max_epochs==3;model.to('cuda:0');assert all(p.device.type=='cuda' for p in model.cv_alg_interface_.model.parameters())
                with threadpool_limits(6):raw_pred=predict_raw(model,V);again=predict_raw(model,V.iloc[:2048])
                repeat=float(np.max(np.abs(raw_pred[:2048]-again)));assert repeat<1e-7;pred=calibrate(raw_pred,calibration['bias']);np.savez_compressed(cache,id=test.id.to_numpy(),raw_pred=raw_pred,pred=pred);new_predictions+=1;dump(receipt,dict(status='complete',completed_utc=now(),cache_sha256=sha(cache),final_plan_sha256=sha(ROOT/'research_r68_final_inference_plan.json'),model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],calibration_sha256=rec['calibration_sha256'],native_members=32,external_query_chunk=2048,repeat_error=repeat,peak_gpu_process_bytes=peak[0]));del model;gc.collect();torch.cuda.empty_cache()
            assert pred.shape==(299844,) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();total+=pred;records.append(dict(fold=k,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],calibration_sha256=rec['calibration_sha256'],prediction_sha256=sha(cache),receipt_sha256=sha(receipt),native_members=32,repeat_error=repeat,seconds=time.monotonic()-started));dump(out/'state.json',dict(status='predicting',completed_models=len(records),updated_utc=now()));print('PREDICTED',k,flush=True);del tr,V;gc.collect()
        neural=total/5;old=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv');tree=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');baselineNN=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(old.id,test.id) and np.array_equal(tree['id'],test.id) and np.array_equal(baselineNN['id'],test.id) and np.max(np.abs(pool(tree['newtree'],baselineNN['newNN'])-old.satisfaction))<1e-14
        a=np.clip(old.satisfaction.to_numpy(),1e-6,1-1e-6);b=np.clip(neural,1e-6,1-1e-6);candidate=expit(.875*logit(a)+.125*logit(b));numerator=a**.875*b**.125;independent=numerator/(numerator+(1-a)**.875*(1-b)**.125);formula_error=float(np.max(np.abs(candidate-independent)));assert formula_error<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v11=old.satisfaction.to_numpy(),new_calibrated_NN=neural,candidate=candidate)
        csv=ROOT/'submission_pure_ranknet_r68_v12.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert list(check)==list(sample) and len(check)==299844 and check.id.is_unique and np.array_equal(check.id,test.id) and roundtrip<1e-14 and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all()
        result=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class='True=1',models=records,native_new_test_members=160,new_classifier_fits=0,new_GPU_model_predictions=new_predictions,formula='Perfold native32meanprob -> owntraininglogitintercept -> meanfivecalibratedprobs -> fixed.875actualv11/.125newexpertlogitpool,clip1e-6',CSV_roundtrip_error=roundtrip,formula_error=formula_error,pre_inference_audit_sha256=sha(ROOT/'research_r68_pre_inference_audit.json'),main_plan_sha256=sha(ROOT/'research_r68_plan.json'),final_plan_sha256=sha(ROOT/'research_r68_final_inference_plan.json'),test_components_sha256=sha(out/'test_components.npz'),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,validation=audit['validation'],new_formal_submissions_since_v11=0,current_daily_quota=None,quota_recheck_before_upload=True,parent_only_submitter=True,actual_top500_verified=False,limitation=plan['statistics_limitation']);assert peak[0]<=CEILING and result['peak_rss_bytes']<=8589934592;dump(out/'result.json',result);dump(ROOT/'research_r68_delivery_audit.json',result);print('CHECKED_CSV',csv.name,result['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as e:dump(ROOT/'research_r68_csv_failure.json',dict(error=repr(e),utc=now()));raise

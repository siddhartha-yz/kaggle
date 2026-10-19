"""Infer the qualified fixed six-view ensemble using cached/saved models, serial GPU1GiB."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_common import ROOT,now,dump,sha
from research_r23_loader import pool
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
import numpy as np,pandas as pd,torch,cloudpickle,joblib,json,time,gc,threading,resource,importlib
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit

def main():
    final=json.loads((ROOT/'research_r64_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r64_plan.json').read_text());audit=json.loads((ROOT/'research_r64_pre_inference_audit.json').read_text())
    assert final['main_plan_sha256']==sha(ROOT/'research_r64_plan.json') and audit['status']=='passed' and audit['verified_model_records']==48 and audit['verified_native_neural_members']==1536
    for n,h in {**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r64_viewbag_{stage}/result.json').read_text())['eligible']
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r64_viewbag_full';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert len(test)==299844 and test.id.is_unique and np.array_equal(test.id,sample.id) and raw.shape[1]==21
        tree=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_full/test_components.npz');old=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');official=pd.read_csv(ROOT/'submission_conditional_neural_r23_v11.csv')
        assert np.array_equal(tree['id'],test.id) and np.array_equal(old['id'],test.id) and np.array_equal(official.id,test.id) and np.max(np.abs(pool(tree['newtree'],old['newNN'])-official.satisfaction))<1e-14
        means=[];records=[]
        for member in plan['members']:
            name=member['name']
            if name=='r23':means.append(old['newNN'].astype('float64'));records.append(dict(view=name,reused_qualified_full_native160=True,component_sha256=sha(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz')));continue
            aux=importlib.import_module(member['loader_module']).auxiliary(test.id,test=True);total=np.zeros(len(test),'float64')
            for k in range(5):
                src=ROOT/'research_artifacts'/f"{member['directory']}_fivefold"/f'fold{k}';rec=json.loads((src/'result.json').read_text());assert rec['status']=='complete' and rec['config']==plan['config'] and rec['model_sha256']==sha(src/'model.pkl') and rec['transformer_sha256']==sha(src/'transformer.joblib')
                cache=out/f'{name}_fold{k}.npz';tr=joblib.load(src/'transformer.joblib');V=tr.transform(raw,aux);assert V.shape==(299844,74);started=time.monotonic()
                if cache.exists():
                    z=np.load(cache);assert np.array_equal(z['id'],test.id) and str(z['model_sha256'])==rec['model_sha256'] and str(z['transformer_sha256'])==rec['transformer_sha256'] and str(z['final_plan_sha256'])==sha(ROOT/'research_r64_final_inference_plan.json');pred=z['pred'];repeat=None
                else:
                    torch.cuda.reset_peak_memory_stats();assert torch.cuda.mem_get_info()[0]>=1024**3
                    with open(src/'model.pkl','rb') as f:model=cloudpickle.load(f)
                    assert np.array_equal(model.classes_,[0,1]) and model.cv_alg_interface_.model.config['n_ens']==32 and model.cv_alg_interface_.model.progress.max_epochs==3
                    model.to('cuda:0');assert all(p.device.type=='cuda' for p in model.cv_alg_interface_.model.parameters())
                    with threadpool_limits(6):pred=model.predict_proba(V)[:,1];again=model.predict_proba(V.iloc[:2048])[:,1]
                    repeat=float(np.max(np.abs(again-pred[:2048])));assert repeat<1e-7
                    np.savez_compressed(cache,id=test.id.to_numpy(),pred=pred,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],final_plan_sha256=sha(ROOT/'research_r64_final_inference_plan.json'))
                    del model;gc.collect();torch.cuda.empty_cache()
                assert pred.shape==(299844,) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();total+=pred;records.append(dict(view=name,fold=k,native_members=32,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256'],prediction_sha256=sha(cache),repeat_error=repeat,seconds=time.monotonic()-started,peak_gpu_process_bytes=peak[0]));dump(out/'state.json',dict(status='predicting',completed_models=len(records),updated_utc=now()));print('PREDICTED',name,k,flush=True);del tr,V;gc.collect()
            means.append(total/5);del aux;gc.collect()
        assert len(means)==6;neural=np.mean(means,axis=0);candidate=pool(tree['newtree'],neural);equivalent=expit(.5*logit(np.clip(tree['newtree'],1e-6,1-1e-6))+.5*logit(np.clip(neural,1e-6,1-1e-6)));error=float(np.max(np.abs(candidate-equivalent)));assert error<1e-12 and np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),v10tree=tree['newtree'],view_neural_means=np.stack(means),newNN=neural,oldNN=old['newNN'],candidate=candidate)
        csv=ROOT/'submission_uniform_viewbag_r64_v12.csv'
        if not csv.exists():pd.DataFrame({'id':test.id,'satisfaction':candidate}).to_csv(csv,index=False,float_format='%.17g')
        check=pd.read_csv(csv);roundtrip=float(np.max(np.abs(check.satisfaction-candidate)));assert list(check)==list(sample) and len(check)==299844 and np.array_equal(check.id,test.id) and check.id.is_unique and roundtrip<1e-14 and np.isfinite(check.satisfaction).all() and check.satisfaction.between(0,1).all()
        report=dict(status='passed',completed_utc=now(),csv=csv.name,sha256=sha(csv),bytes=csv.stat().st_size,rows=len(check),positive_class=True,models=records,native_test_members=960,formula='Arithmetic mean of six NN view probabilities, each mean5native32; unchanged actualv11fulltree equal clipped1e-6 logit pool',CSV_roundtrip_error=roundtrip,formula_error=error,test_components_sha256=sha(out/'test_components.npz'),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,pre_inference_audit_sha256=sha(ROOT/'research_r64_pre_inference_audit.json'),plan_sha256=sha(ROOT/'research_r64_plan.json'),final_plan_sha256=sha(ROOT/'research_r64_final_inference_plan.json'),validation=audit['validation'],new_formal_submissions=0,parent_only_submitter=True,quota_recheck_before_upload=True,actual_top500_verified=False,limitation=plan['statistics_limitation']+' OOF192members/test960 changes inference variance. Original SAT conditions source auxiliary classifiers.')
        assert 0<peak[0]<=CEILING;dump(out/'result.json',report);dump(ROOT/'research_r64_delivery_audit.json',report);print('CHECKED_CSV',csv.name,report['sha256'],flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':
    try:main()
    except BaseException as error:dump(ROOT/'research_r64_csv_failure.json',dict(error=repr(error),updated_utc=now()));raise

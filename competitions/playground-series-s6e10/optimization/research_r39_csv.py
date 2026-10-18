"""Bounded15savedmodel inference pluscachedseed42; uniform640members,no upload."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from research_w3_common import ROOT,now,dump,sha
from research_r23_loader import auxiliary,pool
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING
import gc,threading,resource,json,time,cloudpickle,joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits

def main():
    final=json.loads((ROOT/'research_r39_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r39_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r39_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r39_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['new_model_records_verified']==24 and audit['new_native_members_verified']==768 and audit['final_test_members']==640
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'research_artifacts/r39_uniform_nn_full';out.mkdir(exist_ok=True);peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start()
    try:
        test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert raw.shape==(299844,21) and test.id.is_unique and np.array_equal(test.id,sample.id);aux=auxiliary(test.id,test=True)
        old=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(old['id'],test.id);formal=pool(old['v10tree'],old['newNN']);assert np.max(np.abs(formal-old['candidate']))<1e-12
        previous=ROOT/'submission_conditional_neural_r23_v11.csv';assert sha(previous)=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf';csv=pd.read_csv(previous);assert np.array_equal(csv.id,test.id) and np.max(np.abs(csv.satisfaction.to_numpy()-formal))<1e-14
        blocks=[old['newNN']];records=[]
        with threadpool_limits(6):
            for seed in [43,44,45]:
                total=np.zeros(len(test),'float64')
                for k in range(5):
                    src=ROOT/f'research_artifacts/r39_seed{seed}_lean_fivefold/fold{k}';r=json.loads((src/'result.json').read_text());assert r['status']=='complete' and r['seed']==seed and r['config']==dict(plan['config'],random_state=seed) and r['model_sha256']==sha(src/'model.pkl') and r['transformer_sha256']==sha(src/'transformer.joblib')
                    cache=out/f'seed{seed}_fold{k}.npz';tr=joblib.load(src/'transformer.joblib');V=tr.transform(raw,aux);assert V.shape==(299844,74)
                    with open(src/'model.pkl','rb') as f:m=cloudpickle.load(f)
                    m.to('cuda:0');assert np.array_equal(m.classes_,[0,1]);module=m.cv_alg_interface_.model;assert module.config['n_ens']==32 and module.progress.max_epochs==3
                    if cache.exists():
                        z=np.load(cache);assert np.array_equal(z['id'],test.id);p=z['pred'];replay=m.predict_proba(V.iloc[:2048])[:,1];resume_error=float(np.max(np.abs(p[:2048]-replay)));assert resume_error<1e-7
                    else:
                        p=m.predict_proba(V)[:,1];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.savez_compressed(cache,id=test.id.to_numpy(),pred=p);resume_error=None
                    del m,module;gc.collect();torch.cuda.empty_cache()
                    with open(src/'model.pkl','rb') as f:restored=cloudpickle.load(f)
                    restored.to('cuda:0');replay=restored.predict_proba(V.iloc[:2048])[:,1];err=float(np.max(np.abs(p[:2048]-replay)));assert err<1e-7;total+=p/5
                    records.append(dict(seed=seed,fold=k,model_sha256=r['model_sha256'],prediction_sha256=sha(cache),reload_error=err,resume_error=resume_error));print('TEST_MODEL',seed,k,flush=True);del restored,tr,V;gc.collect();torch.cuda.empty_cache()
                blocks.append(total)
        ensemble=np.mean(blocks,axis=0);candidate=pool(old['v10tree'],ensemble);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
        np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),tree=old['v10tree'],seed42=blocks[0],seed43=blocks[1],seed44=blocks[2],seed45=blocks[3],NN640=ensemble,candidate=candidate,reference_v11=formal)
        destination=ROOT/'submission_uniform_neural_r39_v12.csv';assert not destination.exists();frame=pd.DataFrame(dict(id=test.id.to_numpy(),satisfaction=candidate));assert list(frame)==list(sample);frame.to_csv(destination,index=False,float_format='%.17g');re=pd.read_csv(destination);roundtrip=float(np.max(np.abs(re.satisfaction.to_numpy()-candidate)));assert roundtrip<1e-14 and np.array_equal(re.id,test.id) and len(re)==299844 and re.satisfaction.between(0,1).all();assert 0<peak[0]<=CEILING
        result=dict(status='complete',completed_utc=now(),csv=destination.name,rows=len(re),bytes=destination.stat().st_size,sha256=sha(destination),positive_class='True=1',test_native_members=640,old_seed42_members_reused=160,new_test_models=15,new_test_members=480,fixed_formula='Equal logpool unchangedtree + uniformmeanNN640; all4seedblocks42,43,44,45equallyweighted',plan_sha256=sha(ROOT/'research_r39_plan.json'),final_plan_sha256=sha(ROOT/'research_r39_final_inference_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r39_pre_inference_audit.json'),test_components_sha256=sha(out/'test_components.npz'),model_records=records,CSV_roundtrip_error=roundtrip,all_model_reloads_verified=True,old_v11_exact_hash_and_formula_verified=True,IDs_schema_probabilities_verified=True,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_formal_submissions=0,parent_only_submitter=True,remaining_daily_UI_quota_observed=6,quota_recheck_required=True,actual_top500_verified=False)
        dump(out/'result.json',result);dump(ROOT/'research_r39_delivery_audit.json',result);print('CSV_READY',result['csv'],result['sha256'],flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

"""Audited five-saved-model TabM80-member inference, frozen v11 components, no upload."""
from research_r32_engine import *
from research_r23_loader import auxiliary,pool
from research_r32_check import combine
import json,joblib,pandas as pd,hashlib
from threadpoolctl import threadpool_limits

def main():
    final=json.loads((ROOT/'research_r32_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r32_plan.json').read_text())
    assert final['main_plan_sha256']==sha(ROOT/'research_r32_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audit=json.loads((ROOT/'research_r32_pre_inference_audit.json').read_text());assert audit['status']=='passed' and audit['verified_new_outer_models']==8 and audit['verified_nested_selection_models']==8 and audit['verified_native_members']==128
    assert audit['plan_sha256']==sha(ROOT/'research_r32_plan.json')
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r32_tabm74_{stage}/result.json').read_text())['eligible']
    out=ROOT/'research_artifacts/r32_conditional_tabm_full';out.mkdir(exist_ok=False);peak,stop,watcher=setup(out)
    try:
        with threadpool_limits(6):
            test=pd.read_csv(ROOT/'data/test.csv');sample=pd.read_csv(ROOT/'data/sample_submission.csv');raw=test.drop(columns='id');assert raw.shape==(299844,21) and test.id.is_unique and np.array_equal(test.id,sample.id)
            aux=auxiliary(test.id,test=True);old=np.load(ROOT/'research_artifacts/r23_conditional_nn_full/test_components.npz');assert np.array_equal(old['id'],test.id)
            v11=pool(old['v10tree'],old['newNN']);assert np.max(np.abs(v11-old['candidate']))<1e-12
            oldcsv=ROOT/'submission_conditional_neural_r23_v11.csv';assert sha(oldcsv)=='cc134eb8d9272cd1fd7085fa7f5a5dc84455eb37270c57332f42014a94af75cf';prev=pd.read_csv(oldcsv);assert np.array_equal(prev.id,test.id) and np.max(np.abs(prev.satisfaction-v11))<1e-14
            total=np.zeros(len(test),'float64');records=[]
            for k in range(5):
                run=ROOT/f'research_artifacts/r32_tabm74_fivefold/fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['model_sha256']==sha(run/'model.pt') and r['transformer_sha256']==sha(run/'transformer.joblib')
                tr=joblib.load(run/'transformer.joblib');V=tr.transform(raw,aux);m=load(run/'model.pt');p=predict(m,V);del m;gc.collect();torch.cuda.empty_cache();restored=load(run/'model.pt');q=(V[0][:2048],V[1][:2048]);again=predict(restored,q);error=float(np.max(np.abs(p[:2048]-again)));assert error<1e-7
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.savez_compressed(out/f'fold{k}.npz',id=test.id.to_numpy(),pred=p);total+=p/5
                records.append(dict(fold=k,model_sha256=r['model_sha256'],test_prediction_sha256=sha(out/f'fold{k}.npz'),model_reload_error=error,selected_epochs=r['selected_epochs']));del restored,tr,V;gc.collect();torch.cuda.empty_cache()
            candidate=combine(old['v10tree'],old['newNN'],total);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all()
            np.savez_compressed(out/'test_components.npz',id=test.id.to_numpy(),tree=old['v10tree'],NN74=old['newNN'],TabM74=total,candidate=candidate,reference_v11=v11)
            destination=ROOT/'submission_conditional_tabm_r32_v12.csv';assert not destination.exists();frame=pd.DataFrame(dict(id=test.id.to_numpy(),satisfaction=candidate));assert list(frame)==list(sample);frame.to_csv(destination,index=False,float_format='%.17g')
            re=pd.read_csv(destination);roundtrip=float(np.max(np.abs(re.satisfaction.to_numpy()-candidate)));assert roundtrip<1e-14 and len(re)==299844 and np.array_equal(re.id,test.id) and list(re)==['id','satisfaction'] and np.isfinite(re.satisfaction).all() and re.satisfaction.between(0,1).all()
            assert 0<peak[0]<=1073741824
            result=dict(status='complete',completed_utc=now(),csv=destination.name,rows=len(re),bytes=destination.stat().st_size,sha256=sha(destination),positive_class='True=1',plan_sha256=sha(ROOT/'research_r32_plan.json'),final_plan_sha256=sha(ROOT/'research_r32_final_inference_plan.json'),pre_inference_audit_sha256=sha(ROOT/'research_r32_pre_inference_audit.json'),test_components_sha256=sha(out/'test_components.npz'),native_TabM_members=80,NN74_members_unchanged=160,fixed_formula='sigmoid(.5logit(currenttree)+.375logit(NN74mean160)+.125logit(TabM74mean80)); clip1e-6',models=records,roundtrip_error=roundtrip,peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,formal_new_submissions=0,parent_only_submitter=True,actual_top500_verified=False,remaining_daily_quota_before_any_new_upload=6,quota_recheck_required=True)
            dump(out/'result.json',result);dump(ROOT/'research_r32_delivery_audit.json',dict(result,full_test_used_for_fit=False,strict_ID_order=True,schema_checked=True,old_v11_exact_hash_and_prediction_reconstructed=True,all_model_reloads_checked=True));print(json.dumps(result,indent=2),flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

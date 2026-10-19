"""Label-free native32 uncertainty measurement; observe outputs without changing prediction."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,types,time,gc,threading,cloudpickle,joblib,resource
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT,now,dump,sha
from research_r23_loader import auxiliary
from research_r23_guard import watch
from research_r8_probe import CAP,CEILING

def main():
    plan=json.loads((ROOT/'research_r39_diagnostic_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    torch.set_num_threads(6);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,ROOT/'research_r39_variance_resource_failure.json'),daemon=True);watcher.start()
    try:
        cols=list(pd.read_csv(ROOT/'data/train.csv',nrows=0));df=pd.read_csv(ROOT/'data/train.csv',usecols=[c for c in cols if c!='satisfaction']);raw=df.drop(columns='id');aux=auxiliary(df.id);folds=np.load(ROOT/'research_artifacts/oof_v1/folds.npz');assert np.array_equal(folds['id'],df.id);fold=folds['fold']
        current=np.load(ROOT/'research_artifacts/r23_conditional_lean_fivefold/oof.npz');assert np.array_equal(current['id'],df.id);rows=[];varsum=0.;blendvarsum=0.;N=0
        with threadpool_limits(6):
            for k in range(5):
                vi=np.flatnonzero(fold==k);take=np.random.default_rng(20261040+k).choice(vi,16384,replace=False);src=ROOT/f'research_artifacts/r23_conditional_lean_fivefold/fold{k}';r=json.loads((src/'result.json').read_text());assert r['model_sha256']==sha(src/'model.pkl') and r['transformer_sha256']==sha(src/'transformer.joblib')
                tr=joblib.load(src/'transformer.joblib');V=tr.transform(raw.iloc[take].reset_index(drop=True),aux.iloc[take].reset_index(drop=True));assert V.shape==(len(take),74)
                with open(src/'model.pkl','rb') as f:model=cloudpickle.load(f)
                model.to('cuda:0');module=model.cv_alg_interface_.model;assert module.config['n_ens']==32;original=module._postprocess_ens_pred;captured=[]
                def observe(self,logits):
                    assert logits.shape[0]==32;captured.append(torch.softmax(logits,dim=-1)[:,:,1].numpy().copy());return original(logits)
                module._postprocess_ens_pred=types.MethodType(observe,module)
                try:p=model.predict_proba(V)[:,1]
                finally:module._postprocess_ens_pred=original
                member=np.concatenate(captured,axis=1);assert member.shape==(32,len(take));error=float(np.max(np.abs(member.mean(0)-p)));control_error=float(np.max(np.abs(p-current['pred'][take])));assert error<2e-6 and control_error<2e-6
                variance=member.var(0,ddof=1);se2=variance/32;c=current['candidate'][take];n=np.clip(current['pred'][take],1e-6,1-1e-6);derivative=.5*c*(1-c)/(n*(1-n));blendse2=derivative**2*se2
                row=dict(fold=k,rows=len(take),rms_member_SD=float(np.sqrt(variance.mean())),rms_native32_mean_SE=float(np.sqrt(se2.mean())),rms_formal_logpool_MC_SE_delta_method=float(np.sqrt(blendse2.mean())),observed_mean_prediction_error=error,saved_control_prediction_error=control_error)
                rows.append(row);varsum+=se2.sum();blendvarsum+=blendse2.sum();N+=len(take);print('VARIANCE',row,flush=True);del model,module,tr,V,member,captured;gc.collect();torch.cuda.empty_cache()
        r=dict(status='complete_predictions_only',completed_utc=now(),official_targets_read=False,quality_scores_computed=False,new_models_fitted=0,rows=N,folds=rows,rms_native32_mean_SE=float(np.sqrt(varsum/N)),rms_formal_logpool_MC_SE_delta_method=float(np.sqrt(blendvarsum/N)),hypothetical32_to128_neural_Brier_MC_variance_reduction=float(.75*varsum/N),hypothetical32_to128_formal_Brier_MC_variance_reduction_delta_method=float(.75*blendvarsum/N),peak_gpu_process_bytes=peak[0],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,plan_sha256=sha(ROOT/'research_r39_diagnostic_plan.json'),assumption='Conditionallyindependentrandominitializationsforfixeddata; delta-methodformalpoolapproximation. Measuresfinite-ensembleMCuncertainty,notlabelnoiseorAUCgain. No seedselection.')
        assert 0<peak[0]<=CEILING;dump(ROOT/'research_r39_variance_result.json',r);print(r,flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

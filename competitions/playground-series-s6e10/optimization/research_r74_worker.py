"""Reuse one frozen R68 ensemble, identify each native32 score origin on exact own-training X."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,threading,time,resource,sys,gc
import numpy as np,pandas as pd,torch,cloudpickle,joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
from research_r23_loader import auxiliary
from research_r70_inputs import training_frame
from research_r70_member_origin import predict_margins,fit_origins,predict_calibrated
import research_resource_guard_v2 as limits
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parent
CAP=1610612736;CEILING=2147483648;limits.CEILING=CEILING
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def state(status,**kwargs):dump(ROOT/'research_r74_state.json',dict(status=status,updated_utc=now(),pid=os.getpid(),**kwargs))
def main(stage,k):
 plan=json.loads((ROOT/'research_r74_plan.json').read_text());ph=sha(ROOT/'research_r74_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r74_member_origins_fivefold/result.json').read_text())['eligible']
 out=ROOT/f'research_artifacts/r74_member_origins_{stage}/fold{k}';out.mkdir(parents=True,exist_ok=True);run=ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/fold{k}'
 if (out/'result.json').exists():
  old=json.loads((out/'result.json').read_text())
  if old['status']=='complete':assert old['plan_sha256']==ph and all(sha(out/n)==h for n,h in old['files_sha256'].items());return
 torch.set_num_threads(16);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+limits.MIN_GPU_FREE and limits.host_available()>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;stop=threading.Event();peak=[0];monitor=threading.Thread(target=limits.watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
 try:
  df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');aux=auxiliary(df.id);base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(base['id'],df.id) and np.array_equal(base['target'],y);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);tr=joblib.load(run/'transformer.joblib');X=training_frame(tr,raw,y,aux,fi);V=tr.transform(raw.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));np.savez_compressed(out/'indices.npz',fi=fi,train_id=df.id.iloc[fi].to_numpy(),vi=vi,valid_id=df.id.iloc[vi].to_numpy())
  with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
  module=model.cv_alg_interface_.model;model.to('cuda:0');module.creator.device_info='cuda:0';state('own_training_native_margin_queries',stage=stage,fold=k,outer_quality_scored=False,new_classifier_fits=0)
  with threadpool_limits(16):train_margin=predict_margins(model,X)
  old=np.load(run/'calibration_training.npz');assert np.array_equal(old['idx'],fi) and np.array_equal(old['id'],df.id.iloc[fi]);training_replay=float(np.max(np.abs(expit(train_margin).mean(0)-old['raw_pred'])));assert training_replay<2e-6;np.save(out/'training_margins.npy',train_margin);bias=fit_origins(train_margin,y[fi]);residual=float(np.max(np.abs(expit(train_margin+bias[:,None]).mean(1)-y[fi].mean())));assert residual<1e-12;dump(out/'origins.json',dict(bias=bias.tolist(),native_members=32,rows=len(fi),training_prior=float(y[fi].mean()),maximum_native_prior_residual=residual,fit_on='Exact own-training strict OOF TE representation of saved R68 model',slope=1));del train_margin,X,old;gc.collect();state('validation_native_margin_queries',stage=stage,fold=k,outer_quality_scored=False,new_classifier_fits=0)
  with threadpool_limits(16):valid_margin=predict_margins(model,V);default=model.predict_proba(V.iloc[:512])[:,1]
  old=np.load(run/'validation.npz');assert np.array_equal(old['idx'],vi) and np.array_equal(old['id'],df.id.iloc[vi]);validation_replay=float(np.max(np.abs(expit(valid_margin).mean(0)-old['raw_pred'])));default_error=float(np.max(np.abs(expit(valid_margin[:,:512]).mean(0)-default)));assert validation_replay<2e-6 and default_error<2e-6 and '_postprocess_ens_pred' not in vars(module);pred=predict_calibrated(valid_margin,bias);np.save(out/'validation_margins.npy',valid_margin);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);files={n:sha(out/n) for n in ['indices.npz','training_margins.npy','validation_margins.npy','origins.json','validation.npz']};rec=dict(status='complete',stage=stage,fold=k,plan_sha256=ph,runner_sha256=sha(__file__),files_sha256=files,source_model_sha256=sha(run/'model.pkl'),source_transformer_sha256=sha(run/'transformer.joblib'),training_native_mean_replay_error=training_replay,validation_native_mean_replay_error=validation_replay,default_prediction_restored_error=default_error,maximum_native_prior_residual=residual,new_classifier_fits=0,training_rows=len(fi),validation_rows=len(vi),native_members=32,seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,completed_utc=now());assert 0<peak[0]<=CEILING and rec['peak_allocated_bytes']<=CAP and rec['peak_RSS_bytes']<=8589934592;dump(out/'result.json',rec);state('fold_complete',stage=stage,fold=k,seconds=rec['seconds']);print(json.dumps(rec,indent=2),flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

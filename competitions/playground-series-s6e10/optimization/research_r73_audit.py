"""Independent own-training encoder, causal joint likelihood, saved-model and statistical audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,threading,resource,gc,sys
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit
from research_r73_density_model import JointDensity,RATINGS,CATS,NUMS,RAW
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_GPU_FREE
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reconstruct(raw,fi,stored):
 vocab=[sorted(set(str(x) for x in raw.iloc[fi][n].dropna())) for n in CATS];assert vocab==stored['vocab'] and stored['raw_columns']==RAW and stored['fit_rows']==len(fi)
 num=np.log1p(raw[NUMS].to_numpy('float64'));own=num[fi];median=np.nanmedian(own,axis=0);filled=np.where(np.isfinite(own),own,median);mean=filled.mean(0);scale=np.sqrt(np.mean((filled-mean)**2,axis=0))+1e-6
 for name,a in [('median',median),('mean',mean),('scale',scale)]:assert np.max(np.abs(a-stored[name]))<1e-12,name
 c=np.column_stack([np.array([{v:k+1 for k,v in enumerate(vs)}.get(str(x),0) for x in raw[n]],dtype='int64') for n,vs in zip(CATS,vocab)]);missing=~np.isfinite(num);n=np.concatenate([np.clip((np.where(missing,median,num)-mean)/scale,-10,10),missing],axis=1).astype('float32');r=raw[RATINGS].to_numpy('int64');return [c,n,r],vocab

def main(stage,k):
 plan=json.loads((ROOT/'research_r73_plan.json').read_text());ap=json.loads((ROOT/'research_r73_audit_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r73_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
 for s in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r73_joint_density_{s}/result.json').read_text())['eligible']
 frame=pd.read_csv(ROOT/'data/train.csv');raw=frame[RAW];y=frame.satisfaction.to_numpy('int64');K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(y),-1,'int8')
 for j,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=j
 base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert all(np.array_equal(base[n],v) for n,v in [('fold',fold),('target',y),('id',frame.id)]);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=ROOT/f'research_artifacts/r73_joint_density_{stage}/fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r73_plan.json') and rec['encoder_sha256']==sha(run/'encoder.json') and rec['fit_indices_sha256']==sha(run/'fit_indices.npz') and rec['prediction_sha256']==sha(run/'validation.npz') and all(sha(run/n)==h for n,h in rec['member_files_sha256'].items());fit=np.load(run/'fit_indices.npz');assert np.array_equal(fit['idx'],fi) and np.array_equal(fit['id'],frame.id.iloc[fi]);arrays,vocab=reconstruct(raw,fi,json.loads((run/'encoder.json').read_text()));dest=ROOT/f'research_artifacts/r73_independent_audit/{stage}_fold{k}';dest.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(16);torch.set_num_interop_threads(1);assert torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;peak=[0];stop=threading.Event();watcher=threading.Thread(target=watch,args=(stop,peak,dest/'resource_failure.json'),daemon=True);watcher.start();records=[]
 try:
  for member in range(4):
   saved=np.load(run/f'member{member}.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],frame.id.iloc[vi]);checkpoint=torch.load(run/f'member{member}.pt',map_location='cpu',weights_only=True);assert checkpoint['cat_sizes']==[len(v) for v in vocab];model=JointDensity(checkpoint['cat_sizes'],**plan['architecture']).eval();model.load_state_dict(checkpoint['state_dict']);tensors=[torch.as_tensor(a[vi[:512]]) for a in arrays]
   with torch.inference_mode():
    joint=model.log_joints(*tensors).numpy();CPU=model.predict_proba(*tensors).numpy();independent=expit(joint[:,1].astype('float64')-joint[:,0].astype('float64'));posterior_error=float(np.max(np.abs(CPU-independent)));CPU_error=float(np.max(np.abs(CPU-saved['pred'][:512])));assert posterior_error<2e-6 and CPU_error<2e-5
    changed=tensors[2].clone();position=6;changed[:,position:]=(changed[:,position:]+1)%6;condition=torch.zeros(512,dtype=torch.long);a=model.density_logits(*tensors,condition);b=model.density_logits(tensors[0],tensors[1],changed,1-condition);assert torch.equal(a[0],b[0]);c=model.density_logits(tensors[0],tensors[1],changed,condition);causal_error=float((a[1][:,:position+1]-c[1][:,:position+1]).abs().max());assert causal_error<1e-6
    model.cuda();gpu=[t.cuda() for t in tensors];prediction=model.predict_proba(*gpu).cpu().numpy();GPU_error=float(np.max(np.abs(prediction-saved['pred'][:512])));repeat_error=float(np.max(np.abs(prediction-model.predict_proba(*gpu).cpu().numpy())));assert GPU_error<2e-5 and repeat_error==0
   records.append(dict(member=member,weights_sha256=sha(run/f'member{member}.pt'),CPU_saved_prediction_error=CPU_error,GPU_saved_prediction_error=GPU_error,GPU_repeat_error=repeat_error,independent_Bayes_error=posterior_error,causal_prefix_error=causal_error));del model,gpu;gc.collect();torch.cuda.empty_cache()
  q=np.load(run/'validation.npz');native=np.stack([np.load(run/f'member{j}.npz')['pred'] for j in range(4)]);assert np.array_equal(np.mean(native,axis=0),q['pred']);report=dict(status='passed',stage=stage,fold=k,audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r73_plan.json'),audit_plan_sha256=sha(ROOT/'research_r73_audit_plan.json'),models=records,own_training_preprocessing=True,split_seed=seed,positive_class='True=1',query_target_input=False,peak_gpu_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);assert peak[0]<=CEILING and report['peak_RSS_bytes']<=8589934592;(dest/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('AUDIT_PASSED',stage,k,flush=True)
 finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

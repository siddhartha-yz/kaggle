"""One own-training joint-density ensemble; no outer scoring or target-conditioned query API."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,threading,resource,sys,gc
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_r73_density_model import Encoder,JointDensity,RAW
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_AVAILABLE,MIN_GPU_FREE
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def state(status,**kwargs):dump(ROOT/'research_r73_state.json',dict(status=status,updated_utc=now(),pid=os.getpid(),**kwargs))
def predict(model,arrays,idx,batch):
 out=[]
 with torch.inference_mode():
  for start in range(0,len(idx),batch):
   take=idx[start:start+batch];tensors=[torch.as_tensor(a[take],device='cuda') for a in arrays];out.append(model.predict_proba(*tensors).cpu().numpy().astype('float64'))
 return np.concatenate(out)
def main(stage,k):
 plan=json.loads((ROOT/'research_r73_plan.json').read_text());plan_sha=sha(ROOT/'research_r73_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r73_joint_density_fivefold/result.json').read_text())['eligible']
 frame=pd.read_csv(ROOT/'data/train.csv');y=frame.satisfaction.to_numpy('int64');raw=frame[RAW];assert frame.satisfaction.dtype==bool and raw.shape==(699635,21)
 base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(base['id'],frame.id) and np.array_equal(base['target'],y);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);run=ROOT/f'research_artifacts/r73_joint_density_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'result.json').exists():
  saved=json.loads((run/'result.json').read_text())
  if saved['status']=='complete':
   assert saved['plan_sha256']==plan_sha and saved['prediction_sha256']==sha(run/'validation.npz') and all(sha(run/n)==h for n,h in saved['member_files_sha256'].items());return
 assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*2**30
 torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic();member_records=[]
 try:
  encoder=Encoder().fit(raw.iloc[fi]);assert encoder.fit_rows==len(fi);arrays=encoder.transform(raw);enc=dict(vocab=encoder.vocab,median=encoder.median.tolist(),mean=encoder.mean.tolist(),scale=encoder.scale.tolist(),fit_rows=encoder.fit_rows,raw_columns=RAW);dump(run/'encoder.json',enc);np.savez_compressed(run/'fit_indices.npz',idx=fi,id=frame.id.iloc[fi].to_numpy());training=[torch.as_tensor(a[fi],device='cuda') for a in arrays];target=torch.as_tensor(y[fi],device='cuda');cat_sizes=[len(v) for v in encoder.vocab];member_pred=[];model_files={}
  for j,seed in enumerate(plan['seeds']):
   receipt=run/f'member{j}.json';weights=run/f'member{j}.pt';prediction=run/f'member{j}.npz';torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=JointDensity(cat_sizes,**plan['architecture']).cuda();losses=[]
   if receipt.exists():
    old=json.loads(receipt.read_text());assert old['plan_sha256']==plan_sha and old['weights_sha256']==sha(weights) and old['prediction_sha256']==sha(prediction) and old['encoder_sha256']==sha(run/'encoder.json');checkpoint=torch.load(weights,map_location='cuda',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval();cached=np.load(prediction);assert np.array_equal(cached['idx'],vi) and np.array_equal(cached['id'],frame.id.iloc[vi]);pv=cached['pred'].copy();assert np.max(np.abs(predict(model,arrays,vi[:512],512)-pv[:512]))<1e-6;member_records.append(old)
   else:
    opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay']);rng=np.random.default_rng(seed)
    with threadpool_limits(16):
     for epoch in range(plan['epochs']):
      model.train();order=rng.permutation(len(fi));loss_sum=0.;seen=0
      for start in range(0,len(fi),plan['batch_size']):
       take=torch.as_tensor(order[start:start+plan['batch_size']],device='cuda');opt.zero_grad(set_to_none=True);loss=model.joint_loss(*[a[take] for a in training],target[take]);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);assert torch.isfinite(norm);opt.step();loss_sum+=float(loss.detach())*len(take);seen+=len(take)
      losses.append(loss_sum/seen);state('training',stage=stage,fold=k,member=j,epoch=epoch+1,epochs=plan['epochs'],own_training_joint_NLL=losses[-1],seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],outer_quality_scored=False);print('EPOCH',stage,k,j,epoch+1,losses[-1],flush=True)
    del opt;model.eval();pv=predict(model,arrays,vi,plan['query_batch_size']);assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();reference=predict(model,arrays,vi[:512],512);batch_error=float(np.max(np.abs(pv[:512]-reference)));assert batch_error<2e-5
    torch.save(dict(state_dict={n:t.detach().cpu() for n,t in model.state_dict().items()},cat_sizes=cat_sizes),weights);model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True)['state_dict']);reload_error=float(np.max(np.abs(reference-predict(model,arrays,vi[:512],512))));assert reload_error==0;np.savez_compressed(prediction,idx=vi,id=frame.id.iloc[vi].to_numpy(),pred=pv)
    rec=dict(status='complete',stage=stage,fold=k,member=j,seed=seed,plan_sha256=plan_sha,weights_sha256=sha(weights),prediction_sha256=sha(prediction),encoder_sha256=sha(run/'encoder.json'),own_training_joint_NLL_by_epoch=losses,batch_invariance_error=batch_error,saved_GPU_reload_error=reload_error,training_rows=len(fi),validation_rows=len(vi),query_target_input=False,completed_utc=now());dump(receipt,rec);member_records.append(rec)
   member_pred.append(pv);model_files[weights.name]=sha(weights);model_files[prediction.name]=sha(prediction);model_files[receipt.name]=sha(receipt);del model;gc.collect();torch.cuda.empty_cache()
  mean=np.mean(np.stack(member_pred),axis=0);np.savez_compressed(run/'validation.npz',idx=vi,id=frame.id.iloc[vi].to_numpy(),target=y[vi],pred=mean);rec=dict(status='complete',stage=stage,fold=k,plan_sha256=plan_sha,runner_sha256=sha(__file__),encoder_sha256=sha(run/'encoder.json'),fit_indices_sha256=sha(run/'fit_indices.npz'),prediction_sha256=sha(run/'validation.npz'),member_files_sha256=model_files,members=member_records,seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,completed_utc=now());assert 0<peak[0]<=CEILING and rec['peak_allocated_bytes']<=CAP and rec['peak_RSS_bytes']<=8589934592;dump(run/'result.json',rec);state('fold_complete',stage=stage,fold=k,seconds=rec['seconds']);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

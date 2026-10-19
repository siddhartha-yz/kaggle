"""One fixed Mitra-v2 outer fold: own labels only, four 50-step adaptations."""
from research_r76_model import ROOT,RawEncoder,make_model,logits,predict,parameter_digest,balanced_context,save_file,SupportCache
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
import os,json,hashlib,time,threading,resource,gc,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def state(status,**kw):
 d=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),outer_quality_scored=False,**kw);dump(ROOT/'research_r76_state.json',d);print(json.dumps(d),flush=True)
def save_npz(path,**arrays):
 if path.exists():
  with np.load(path) as old:assert set(old.files)==set(arrays) and all(np.array_equal(old[n],a) for n,a in arrays.items())
  return
 t=path.with_name('.'+path.stem+f'.{os.getpid()}.npz');np.savez_compressed(t,**arrays)
 try:os.link(t,path)
 finally:t.unlink(missing_ok=True)

def main(stage,k):
 plan=json.loads((ROOT/'research_r76_plan.json').read_text());ph=sha(ROOT/'research_r76_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r76_mitra_v2_fivefold/result.json').read_text())['eligible']
 run=ROOT/f'research_artifacts/r76_mitra_v2_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'result.json').exists():
  rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==ph and all(sha(run/n)==h for n,h in rec['files_sha256'].items());return
 torch.set_num_threads(16);torch.set_num_interop_threads(1)
 assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=6*2**30
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');assert len(frame)==699635 and frame.satisfaction.dtype==bool and frame.id.is_unique
  ids=frame.id.to_numpy(copy=True);y=frame.satisfaction.to_numpy('int64');raw=frame.drop(columns=['id','satisfaction']);base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(ids,base['id']) and np.array_equal(y,base['target'])
  fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);encoder=RawEncoder().fit(raw.iloc[fi]);X=encoder.transform(raw.iloc[fi]);V=encoder.transform(raw.iloc[vi]);dump(run/'encoder.json',encoder.state());save_npz(run/'outer_indices.npz',fi=fi,vi=vi,train_id=ids[fi],valid_id=ids[vi]);del frame,raw,encoder;gc.collect()
  native_predictions=[];hashes={'encoder.json':sha(run/'encoder.json'),'outer_indices.npz':sha(run/'outer_indices.npz')}
  for seed in plan['training']['seeds']:
   own=run/f'native{seed}';own.mkdir(exist_ok=True);metadata=own/'training.json';ckpt=own/'model.safetensors';torch.manual_seed(seed);np.random.seed(seed)
   if metadata.exists():
    fit=json.loads(metadata.read_text())
    if fit['plan_sha256']!=ph:
     legacy=plan['engineering_recovery']['accepted_training'][f'{stage}/fold{k}/native{seed}'];assert fit['plan_sha256']==legacy['plan_sha256'] and sha(metadata)==legacy['training_metadata_sha256'] and fit['files_sha256']==legacy['files_sha256']
    assert fit['seed']==seed and fit['completed_steps']==50 and all(sha(own/n)==h for n,h in fit['files_sha256'].items())
    model=make_model(checkpoint=ckpt);assert parameter_digest(model)==fit['parameters_sha256'];state('restored_completed_native',stage=stage,fold=k,seed=seed)
   else:
    model=make_model();assert parameter_digest(model)==plan['pretrained_parameters_sha256'];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.3,foreach=False);rng=np.random.default_rng(seed);episodes=[];losses=[]
    for step in range(50):
     selected=rng.choice(len(fi),1024,replace=False);s,q=selected[:512],selected[512:];assert len(np.intersect1d(s,q))==0
     episodes.append(selected);lr=1e-5*min(1.,(step+1)/10)
     for group in optimizer.param_groups:group['lr']=lr
     optimizer.zero_grad(set_to_none=True);model.train();out=logits(model,X[s],y[fi[s]],X[q]);target=torch.tensor(y[fi[q]],device='cuda',dtype=torch.long);loss=torch.nn.functional.cross_entropy(out,target)
     assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step();losses.append(float(loss.detach()))
     del out,loss,target
     if (step+1)%10==0:state('microtuning',stage=stage,fold=k,seed=seed,completed_steps=step+1,training_loss=losses[-1],peak_GPU_bytes=peak[0],seconds=time.monotonic()-tick)
    optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache();model.eval();trained=parameter_digest(model);assert trained!=plan['pretrained_parameters_sha256'];torch.cuda.synchronize()
    cpu={n:v.detach().cpu().contiguous() for n,v in model.state_dict().items()};temp=own/'.model.safetensors.tmp';save_file(cpu,str(temp));del cpu;os.replace(temp,ckpt)
    ep=np.asarray(episodes);save_npz(own/'episodes.npz',own_positions=ep,global_idx=fi[ep],global_id=ids[fi[ep]],training_loss=np.asarray(losses,dtype='float64'))
    fit=dict(plan_sha256=ph,seed=seed,completed_steps=50,own_training_rows=len(fi),parameters_sha256=trained,initial_parameters_sha256=plan['pretrained_parameters_sha256'],outer_validation_used=False,loss='Official ten-class query crossentropy, labels0/1',files_sha256={n:sha(own/n) for n in ['model.safetensors','episodes.npz']});dump(metadata,fit)
    # AdamW's last param-group loop variable owns the model parameters, too.
    # Release every training reference before reclaiming fragmented segments.
    del model,group;gc.collect();torch.cuda.empty_cache();model=make_model(checkpoint=ckpt);assert parameter_digest(model)==trained
   ctx=balanced_context(y[fi],4096,np.random.default_rng(np.random.SeedSequence([seed,7601])));save_npz(own/'context.npz',own_positions=ctx,global_idx=fi[ctx],global_id=ids[fi[ctx]],target=y[fi[ctx]])
   reference_first=predict(model,X[ctx],y[fi[ctx]],V[:1024]);torch.cuda.empty_cache();cache=SupportCache(model,X[ctx],y[fi[ctx]])
   prediction=np.full(len(vi),np.nan);resume_error=None;first=None
   for start in range(0,len(vi),1024):
    take=vi[start:start+1024];query=V[start:start+1024];file=own/f'query{start:06d}.npz'
    if file.exists():
     saved=np.load(file);assert str(saved['plan_sha256'])==ph and np.array_equal(saved['idx'],take) and np.array_equal(saved['id'],ids[take]);pv=saved['pred']
     if resume_error is None:resume_error=float(np.max(np.abs(pv-cache.predict(query))));assert resume_error<1e-6
    else:
     pv=cache.predict(query);save_npz(file,idx=take,id=ids[take],pred=pv,plan_sha256=np.asarray(ph))
    assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();prediction[start:start+len(take)]=pv
    if first is None:first=pv.copy()
    if start%(16*1024)==0 or start+1024>=len(vi):state('query_prediction',stage=stage,fold=k,seed=seed,predicted_rows=min(start+1024,len(vi)),valid_rows=len(vi),peak_GPU_bytes=peak[0],seconds=time.monotonic()-tick)
   repeat=float(np.max(np.abs(first-cache.predict(V[:len(first)]))));first_error=float(np.max(np.abs(first-reference_first)));del cache;gc.collect();torch.cuda.empty_cache();last_error=float(np.max(np.abs(prediction[start:]-predict(model,X[ctx],y[fi[ctx]],V[start:]))));assert max(repeat,first_error,last_error)<1e-6 and parameter_digest(model)==fit['parameters_sha256'];save_npz(own/'validation.npz',idx=vi,id=ids[vi],pred=prediction)
   done=dict(status='complete',seed=seed,plan_sha256=ph,training_metadata_sha256=sha(metadata),repeat_error=repeat,resume_error=resume_error,cache_first1024_error=first_error,cache_last_query_error=last_error,parameters_unchanged_during_inference=True,files_sha256={n:sha(own/n) for n in ['context.npz','validation.npz']});dump(own/'result.json',done);hashes[f'native{seed}/result.json']=sha(own/'result.json');hashes[f'native{seed}/training.json']=sha(metadata);native_predictions.append(prediction);del model;gc.collect();torch.cuda.empty_cache()
  pooled=np.mean(native_predictions,axis=0);save_npz(run/'validation.npz',idx=vi,id=ids[vi],target=y[vi],pred=pooled);hashes['validation.npz']=sha(run/'validation.npz')
  result=dict(status='complete',stage=stage,fold=k,plan_sha256=ph,runner_sha256=sha(__file__),native_seeds=plan['training']['seeds'],files_sha256=hashes,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,outer_quality_scored=False);dump(run/'result.json',result);state('fold_complete',stage=stage,fold=k,seconds=result['seconds'])
 finally:stop.set()
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

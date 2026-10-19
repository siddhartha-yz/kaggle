"""One frozen raw21 TabDPT fold: own preprocessing/context labels, immutable pretrained weights."""
from research_r75_model import ROOT,RawEncoder,make_model,predict,parameter_digest
import os,json,hashlib,time,threading,resource,gc,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_GPU_FREE
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);temp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,p)
def state(status,**kw):dump(ROOT/'research_r75_state.json',dict(status=status,updated_utc=now(),pid=os.getpid(),pretrained_weight_updates=False,**kw))
def save_npz(path,**arrays):
 if path.exists():
  with np.load(path) as old:assert set(old.files)==set(arrays) and all(np.array_equal(old[n],a) for n,a in arrays.items())
  return
 temp=path.with_name('.'+path.stem+f'.{os.getpid()}.npz');np.savez_compressed(temp,**arrays)
 try:os.link(temp,path)
 finally:temp.unlink(missing_ok=True)
def main(stage,k):
 plan=json.loads((ROOT/'research_r75_plan.json').read_text());ph=sha(ROOT/'research_r75_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 if stage=='threefold':assert json.loads((ROOT/'research_artifacts/r75_tabdpt_raw21_fivefold/result.json').read_text())['eligible']
 run=ROOT/f'research_artifacts/r75_tabdpt_raw21_{stage}/fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'result.json').exists():
  old=json.loads((run/'result.json').read_text())
  if old['status']=='complete':assert old['plan_sha256']==ph and all(sha(run/n)==h for n,h in old['files_sha256'].items());return
 torch.set_num_threads(16);torch.set_num_interop_threads(1);available=torch.cuda.is_available();free=torch.cuda.mem_get_info()[0] if available else 0;host=host_available();print('PREFLIGHT',dict(CUDA=available,GPU_free=free,host_available=host),flush=True);assert available and free>=CAP+MIN_GPU_FREE and host>=6*2**30;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');assert frame.satisfaction.dtype==bool and len(frame)==699635 and frame.id.is_unique;ids=frame.id.to_numpy(copy=True);y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(base['id'],ids) and np.array_equal(base['target'],y);fi=np.flatnonzero(base['fold']!=k);vi=np.flatnonzero(base['fold']==k);assert np.array_equal(np.unique(y[fi]),[0,1]);encoder=RawEncoder().fit(raw.iloc[fi]);X=encoder.transform(raw.iloc[fi]);V=encoder.transform(raw.iloc[vi]);assert X.shape==(len(fi),21) and V.shape==(len(vi),21);dump(run/'encoder.json',encoder.state());del frame,raw,encoder;gc.collect();state('own_training_preprocessing',stage=stage,fold=k,outer_quality_scored=False);model=make_model('cuda');before=parameter_digest(model.model);assert before==plan['pretrained_parameters_sha256'];model.fit(X,y[fi]);assert model.num_classes==2 and model.n_instances==len(fi) and model.n_features==21 and model.projection is None and model.use_flash and not model.compile and not model.model.training and np.array_equal(model.y_train,y[fi]);assert sorted({str(p.device) for p in model.model.parameters()})==['cuda:0'];dump(run/'preprocessing.json',dict(imputer_statistics=model.imputer.statistics_.tolist(),scaler_mean=model.scaler.mean_.tolist(),scaler_scale=model.scaler.scale_.tolist(),training_rows=len(fi),feature_width=21))
  context_arrays=dict(fi=fi,train_id=ids[fi],vi=vi,valid_id=ids[vi]);seeds=list(model._get_ensemble_iterator(4,42))
  for j,seed in enumerate(seeds):
   seed=int(seed);selected=model._get_subsample_indices(8192,seed);assert len(np.unique(selected))==8192 and np.all((selected>=0)&(selected<len(fi)));context_arrays[f'context{j}_idx']=fi[selected];context_arrays[f'context{j}_id']=ids[fi[selected]];context_arrays[f'context{j}_seed']=np.asarray(seed,dtype='uint64')
  if (run/'contexts.npz').exists():
   saved=np.load(run/'contexts.npz');assert set(saved.files)==set(context_arrays) and all(np.array_equal(saved[n],a) for n,a in context_arrays.items())
  else:save_npz(run/'contexts.npz',**context_arrays)
  probability=np.full(len(vi),np.nan);resume_error=None;first=None
  for start in range(0,len(vi),4096):
   take=vi[start:start+4096];file=run/f'query{start:06d}.npz';query=V[start:start+4096]
   if file.exists():
    cache=np.load(file);assert str(cache['plan_sha256'])==ph and np.array_equal(cache['idx'],take) and np.array_equal(cache['id'],ids[take]);pv=cache['pred'];assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all()
    if resume_error is None:resume_error=float(np.max(np.abs(pv-predict(model,query))));assert resume_error<1e-6
   else:
    pv=predict(model,query);save_npz(file,idx=take,id=ids[take],pred=pv,plan_sha256=np.asarray(ph))
   probability[start:start+len(take)]=pv
   if first is None:first=pv.copy()
   if start%(4*4096)==0 or start+4096>=len(vi):state('query_prediction',stage=stage,fold=k,predicted_rows=min(start+4096,len(vi)),valid_rows=len(vi),seconds=time.monotonic()-tick,peak_gpu_process_bytes=peak[0],outer_quality_scored=False);print('PROGRESS',stage,k,min(start+4096,len(vi)),time.monotonic()-tick,flush=True)
  repeat=float(np.max(np.abs(first-predict(model,V[:len(first)]))));assert repeat<1e-6 and before==parameter_digest(model.model);assert sha(ROOT/'research_sources/tabdpt/tabdpt1_3.safetensors')==plan['artifact_sha256']['research_sources/tabdpt/tabdpt1_3.safetensors'];save_npz(run/'validation.npz',idx=vi,id=ids[vi],target=y[vi],pred=probability);files={n:sha(run/n) for n in ['encoder.json','preprocessing.json','contexts.npz','validation.npz']};rec=dict(status='complete',stage=stage,fold=k,plan_sha256=ph,runner_sha256=sha(__file__),files_sha256=files,query_chunks=(len(vi)+4095)//4096,training_rows=len(fi),validation_rows=len(vi),feature_width=21,pretrained_parameters_sha256=before,pretrained_weights_unchanged=True,query_target_input=False,own_training_only_preprocessing=True,context_size=8192,native_ensembles=4,repeat_first_full_query_batch_error=repeat,resume_replay_error=resume_error,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,completed_utc=now());assert 0<peak[0]<=CEILING and rec['peak_allocated_bytes']<=CAP and rec['peak_RSS_bytes']<=8589934592;dump(run/'result.json',rec);state('fold_complete',stage=stage,fold=k,seconds=rec['seconds']);print('FOLD_COMPLETE',stage,k,rec['seconds'],flush=True)
 finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

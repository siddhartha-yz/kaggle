"""One complete paired outer fit; holdout SAT labels never parsed by this worker."""
from research_r94_common import *
from research_r94_model import *
from research_resource_guard_v2 import watch,host_available,CAP,MIN_GPU_FREE
from research_resume_policy_20261021 import check_training
from sklearn.model_selection import StratifiedShuffleSplit
import copy,gc,math,resource,signal,sys,time

def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def model_sha(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(tensor_sha(t).encode())
 return h.hexdigest()
def save(path,value):
 reserve();t=path.with_name('.'+path.name+'.tmp');torch.save(value,t);t.replace(path)
def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore_rng(s):torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda'])
def predict(model,n,c,batch=512):
 model.eval();values=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,len(n),batch):
   check_training();values.append(torch.sigmoid(model(n[start:start+batch],c[start:start+batch])[1].float()).cpu().numpy())
 return np.concatenate(values).astype('float32')
def logloss(y,p):
 q=np.clip(p.astype('float64'),1e-7,1-1e-7)
 return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q)))

def main(stage,k):
 p=plan();reserve();check_training();ph=sha(ROOT/'research_r94_full_plan.json')
 run=OUT/stage/f'fold{k}';run.mkdir(parents=True,exist_ok=True)
 if (run/'complete.json').exists():
  old=json.loads((run/'complete.json').read_text());assert old['plan_sha256']==ph
  for n,h in old['files_sha256'].items():assert sha(run/n)==h
  print('SKIP_VERIFIED_COMPLETED_PAIR',stage,k,flush=True);return
 if stage=='threefold':assert json.loads((OUT/'fivefold/result.json').read_text())['eligible']
 started=time.monotonic();torch.set_num_threads(6);torch.set_num_interop_threads(1)
 torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and host_available()>=6*2**30
 assert torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
 torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 stop=threading.Event();peak=[0]
 threading.Thread(target=watch,args=(stop,peak,run/'resource_failure.json'),daemon=True).start()
 def disk_guard():
  while not stop.wait(.5):
   try:reserve();check_training()
   except BaseException as e:dump(run/'resource_failure.json',dict(reason=repr(e)));os._exit(86)
 threading.Thread(target=disk_guard,daemon=True).start()
 def state(status,**kw):
  d=dict(status=status,updated_utc=now(),PID=os.getpid(),stage=stage,fold=k,seconds=time.monotonic()-started,
         peak_process_GPU_bytes=peak[0],outer_quality_scored=False,**kw)
  dump(ROOT/'research_r94_state.json',d);print(json.dumps(d),flush=True)
 try:
  base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz')
  fold=base['fold'];ids=base['id'];fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k)
  assert len(fi)+len(vi)==699635 and k in np.unique(fold)
  raw=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*FEATURES]);assert np.array_equal(raw.id,ids)
  keep=np.zeros(len(raw),dtype=bool);keep[fi]=True
  # Parser skips all outer validation data rows before reading satisfaction values.
  own=pd.read_csv(ROOT/'data/train.csv',usecols=['satisfaction'],skiprows=lambda row:row>0 and not keep[row-1])
  y=own.satisfaction.to_numpy('float32');assert len(y)==len(fi) and set(np.unique(y))=={0.,1.}
  splitter=StratifiedShuffleSplit(1,test_size=.1,random_state=2026102203+k+(100 if stage=='threefold' else 0))
  it,ih=next(splitter.split(fi,y));ti,iv=fi[it],fi[ih]
  assert not np.intersect1d(np.r_[ti,iv],vi).size and not np.intersect1d(ti,iv).size
  pre=Preprocessor().fit(raw.iloc[ti]);n,c=pre.transform(raw.iloc[ti]);vn,vc=pre.transform(raw.iloc[iv])
  # Outer features transformed only after the preprocessor fit; never an RTD donor.
  qn,qc=pre.transform(raw.iloc[vi]);del raw,own
  index_proof=dict(fi_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),vi_sha256=hashlib.sha256(vi.tobytes()).hexdigest(),
    inner_train_sha256=hashlib.sha256(ti.tobytes()).hexdigest(),inner_valid_sha256=hashlib.sha256(iv.tobytes()).hexdigest(),
    own_labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),inner_train_rows=len(ti),inner_valid_rows=len(iv),outer_rows=len(vi))
  prep=dict(plan_sha256=ph,index_proof=index_proof,preprocessor=pre.state(),pretraining_donor_rows='inner_train_only',outer_labels_parsed=False)
  if (run/'preprocessing.json').exists():assert json.loads((run/'preprocessing.json').read_text())==prep
  else:dump(run/'preprocessing.json',prep)
  n,c,vn,vc,qn,qc=[t.cuda() for t in [n,c,vn,vc,qn,qc]];yt=torch.from_numpy(y[it]).cuda();yv=y[ih]
  seed=p['initialization_seed']+k+(100 if stage=='threefold' else 0)
  torch.manual_seed(seed);model=RTDModel(pre).cuda();initial={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
  initial_hash=model_sha(model);predictions={};arm_records={}
  for arm in ['scratch','rtd']:
   complete=run/f'{arm}_complete.json';best=run/f'{arm}_best.pt';resume=run/f'{arm}_resume.pt'
   if complete.exists():
    saved=json.loads(complete.read_text());assert saved['plan_sha256']==ph and saved['initial_hash']==initial_hash
    assert sha(best)==saved['best_sha256'] and sha(resume)==saved['resume_sha256']
    arm_records[arm]=saved;state('reuse_completed_arm',arm=arm)
   else:
    model.load_state_dict(initial)
    records=[];best_loss=float('inf');best_epoch=0;resume_phase=None;completed_epochs=0
    saved_state=None
    if resume.exists():
     saved_state=torch.load(resume,map_location='cpu',weights_only=False)
     assert saved_state['plan_sha256']==ph and saved_state['initial_hash']==initial_hash
     model.load_state_dict(saved_state['model']);records=saved_state['records'];best_loss=saved_state['best_loss'];best_epoch=saved_state['best_epoch']
     resume_phase=saved_state['phase'];completed_epochs=saved_state['completed_epochs']
    for phase,epochs in ([('pretrain',5),('supervised',20)] if arm=='rtd' else [('supervised',20)]):
     if resume_phase=='supervised' and phase=='pretrain':continue
     optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
     scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00001)
     first=0
     if saved_state is not None and resume_phase==phase:
      optimizer.load_state_dict(saved_state['optimizer']);scheduler.load_state_dict(saved_state['scheduler']);restore_rng(saved_state['rng'])
      first=completed_epochs
     for epoch in range(first,epochs):
      check_training();model.train();tick=time.monotonic()
      # Epoch-specific randomness pairs both supervised arms irrespective of RTD history.
      epoch_seed=seed+10000+(1000 if phase=='pretrain' else 0)+epoch
      torch.manual_seed(epoch_seed);order_generator=torch.Generator(device='cuda').manual_seed(epoch_seed+100000)
      corruption_generator=torch.Generator(device='cuda').manual_seed(epoch_seed+200000)
      order=torch.randperm(len(n),generator=order_generator,device='cuda')
      loss_sum=torch.zeros((),device='cuda');changed_sum=torch.zeros((),device='cuda');rows=0;lr=optimizer.param_groups[0]['lr']
      for start in range(0,len(n),512):
       if start%16384==0:check_training()
       ix=order[start:start+512];a,b=n[ix],c[ix]
       if phase=='pretrain':
        selected=torch.rand((len(ix),21),generator=corruption_generator,device='cuda')<.15
        donors=torch.randint(len(n),(len(ix),21),generator=corruption_generator,device='cuda')
        a,b,target=corrupt(a,b,n,c,selected,donors);changed_sum+=target.sum()
       else:target=yt[ix]
       optimizer.zero_grad(set_to_none=True)
       with torch.autocast('cuda',dtype=torch.bfloat16):
        rt,cl=model(a,b);loss=torch.nn.functional.binary_cross_entropy_with_logits(rt if phase=='pretrain' else cl,target)
       loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
       loss_sum+=loss.detach()*len(ix);rows+=len(ix)
      torch.cuda.synchronize();value=float(loss_sum/rows);assert math.isfinite(value)
      rec=dict(phase=phase,epoch=epoch+1,rows=rows,training_loss=value,training_seconds=time.monotonic()-tick,lr=lr,epoch_seed=epoch_seed)
      if phase=='supervised':
       pv=predict(model,vn,vc);iloss=logloss(yv,pv);rec['inner_logloss']=iloss
       if iloss<best_loss:
        best_loss=iloss;best_epoch=epoch+1;save(best,{key:value.detach().cpu() for key,value in model.state_dict().items()})
      else:rec['actual_changed_fraction']=float(changed_sum/(rows*21))
      records.append(rec);scheduler.step()
      save(resume,dict(plan_sha256=ph,initial_hash=initial_hash,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),phase=phase,completed_epochs=epoch+1,records=records,best_loss=best_loss,best_epoch=best_epoch))
      state('epoch_complete',arm=arm,phase=phase,epoch=epoch+1,epoch_seconds=rec['training_seconds'],inner_logloss=rec.get('inner_logloss'))
     # Completed pretraining cannot influence paired supervised dropout/order seeds.
     saved_state=None;resume_phase=None
     del optimizer,scheduler
    assert len([a for a in records if a['phase']=='supervised'])==20
    assert len([a for a in records if a['phase']=='pretrain'])==(5 if arm=='rtd' else 0)
    saved=dict(status='fit_complete',plan_sha256=ph,arm=arm,initial_hash=initial_hash,best_epoch=best_epoch,best_inner_logloss=best_loss,
      best_sha256=sha(best),resume_sha256=sha(resume),records=records,outer_labels_parsed=False,preprocessing_sha256=sha(run/'preprocessing.json'))
    dump(complete,saved);arm_records[arm]=saved
   model.load_state_dict(torch.load(best,map_location='cuda',weights_only=True));model.eval()
   before=model_sha(model);pred=predict(model,qn,qc);again=predict(model,qn[:512],qc[:512])
   assert np.array_equal(pred[:512],again) and before==model_sha(model)
   assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();predictions[arm]=pred
   state('arm_queries_complete',arm=arm,predicted_rows=len(pred),best_inner_epoch=arm_records[arm]['best_epoch'])
  assert arm_records['scratch']['initial_hash']==arm_records['rtd']['initial_hash']
  assert [a['epoch_seed'] for a in arm_records['scratch']['records']]==[a['epoch_seed'] for a in arm_records['rtd']['records'] if a['phase']=='supervised']
  reserve();target=run/'predictions.npz';temp=run/'.predictions.npz';np.savez_compressed(temp,idx=vi.astype('int32'),id=ids[vi],scratch=predictions['scratch'],rtd=predictions['rtd']);temp.replace(target)
  files=['preprocessing.json','scratch_best.pt','scratch_resume.pt','scratch_complete.json','rtd_best.pt','rtd_resume.pt','rtd_complete.json','predictions.npz']
  result=dict(status='complete_paired_outerfit',stage=stage,fold=k,plan_sha256=ph,files_sha256={n:sha(run/n) for n in files},
    outer_labels_parsed=False,outer_quality_scored=False,initial_hash=initial_hash,
    seconds=time.monotonic()-started,peak_process_GPU_bytes=peak[0],peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,artifact_bytes=ownbytes())
  dump(run/'complete.json',result);state('paired_fold_complete',elapsed_seconds=result['seconds'])
 finally:stop.set()

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))

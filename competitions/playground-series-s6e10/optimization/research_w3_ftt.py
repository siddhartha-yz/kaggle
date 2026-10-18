"""Official FT-Transformer, inner-only epoch selection, fresh outer refits and frozen blending."""
from pathlib import Path
import os
os.environ['CUDA_CACHE_PATH']=str(Path(__file__).absolute().parent.parent/'.cache-ftt-w3/cuda')
from research_w3_common import *
import argparse,time,gc,joblib,subprocess
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from rtdl_revisiting_models import FTTransformer
from research_w3_ftt_features import FTTFeatures
from research_external_ablation import external

def cfg():return json.loads((ROOT/'research_w3_ftt_plan.json').read_text())
def own_gpu_mib():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  a=line.split(',')
  if len(a)==2 and a[0].strip()==str(os.getpid()):
   try:return int(a[1].strip())
   except ValueError:pass
 return None
def create(n,cards):return FTTransformer(n_cont_features=int(n),cat_cardinalities=list(cards),d_out=1,**FTTransformer.get_default_kwargs(cfg().get('n_blocks',3)))
def gpu_setup():
 hw=setup_gpu();cap=778043392;torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory);assert torch.cuda.is_bf16_supported();hw['reduced_allocator_cap_bytes']=cap;return hw
def batch(x,ix,device='cuda'):return torch.from_numpy(np.ascontiguousarray(x[0][ix])).to(device),torch.from_numpy(np.ascontiguousarray(x[1][ix])).to(device)
@torch.no_grad()
def predict(m,x,bs=256,device='cuda'):
 m.eval();out=[]
 for begin in range(0,len(x[0]),bs):out.append(torch.sigmoid(m(*batch(x,slice(begin,begin+bs),device)).flatten()).cpu().numpy())
 return np.concatenate(out)
def save_atomic(path,state):
 temp=path.with_name(path.name+'.tmp');torch.save(state,temp);temp.replace(path)
def load_model(path,device='cuda'):
 z=torch.load(path,map_location='cpu',weights_only=True);m=create(z['n_cont'],z['cards']);m.load_state_dict(z['model']);return m.to(device)

def train(x,y,cards,seed,out,epochs,inner=None):
 """Training accepts only outer-training labels; inner argument, if any, is a subset thereof."""
 pp=cfg();out.mkdir(exist_ok=True);identity={'seed':seed,'epochs':epochs,'n_cont':x[0].shape[1],'cards':list(cards),'rows':len(y),'plan_sha256':sha(ROOT/'research_w3_ftt_plan.json'),'batch_size':pp['chosen_batch_rows'],'has_inner':inner is not None};checkpoint=out/'checkpoint.pt';curve=[];best=float('inf');best_epoch=0;bad=0;torch.manual_seed(seed);rng=np.random.default_rng(seed);m=create(x[0].shape[1],cards).to('cuda');opt=torch.optim.AdamW(m.make_parameter_groups(),lr=1e-4,weight_decay=1e-5,foreach=False);first=1
 if checkpoint.exists():
  z=torch.load(checkpoint,map_location='cpu',weights_only=True);assert z['identity']==identity;m.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);torch.set_rng_state(z['torch_rng']);torch.cuda.set_rng_state(z['cuda_rng']);rng.bit_generator.state=z['numpy_rng'];curve=z['curve'];first=z['completed_epoch']+1;best=z['best'];best_epoch=z['best_epoch'];bad=z['bad']
 if inner is not None and bad>=pp['patience']:epochs=first-1
 for epoch in range(first,epochs+1):
  gpu_epoch_guard();started=time.monotonic();m.train();order=rng.permutation(len(y));loss_sum=torch.zeros((),device='cuda');count=0
  for begin in range(0,len(order),pp['chosen_batch_rows']):
   ix=order[begin:begin+pp['chosen_batch_rows']];n,c=batch(x,ix);target=torch.from_numpy(y[ix].astype('float32')).to('cuda');opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(n,c).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
   loss.backward();opt.step();loss_sum+=loss.detach()*len(ix);count+=len(ix)
  torch.cuda.synchronize();used=own_gpu_mib();assert used is None or used<=1024,'Process GPU exceeds1GiB';row={'epoch':epoch,'train_loss':float(loss_sum/count),'seconds':time.monotonic()-started,'gpu_free_bytes':gpu_epoch_guard(),'process_gpu_mib':used,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()};assert np.isfinite(row['train_loss'])
  if inner is not None:
   pv=predict(m,inner[0]);q=np.clip(pv.astype('float64'),1e-7,1-1e-7);iy=inner[1];ll=float(-(iy*np.log(q)+(1-iy)*np.log1p(-q)).mean());row['inner_logloss']=ll
   if ll<best-pp['inner_min_delta']:best=ll;best_epoch=epoch;bad=0
   else:bad+=1
  else:best_epoch=epoch
  curve.append(row);save_atomic(checkpoint,{'identity':identity,'model':m.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':rng.bit_generator.state,'curve':curve,'completed_epoch':epoch,'best':best,'best_epoch':best_epoch,'bad':bad});dump(out/'progress.json',{'status':'running','last_progress_utc':now(),'identity':identity,'curve':curve,'best_epoch':best_epoch,'bad':bad});print(str(out.relative_to(ROOT)),row,flush=True)
  if inner is not None and bad>=pp['patience']:break
 save_atomic(out/'model.pt',{'n_cont':identity['n_cont'],'cards':list(cards),'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'seed':seed,'epochs':len(curve)});dump(out/'progress.json',{'status':'complete','completed_utc':now(),'identity':identity,'curve':curve,'best_epoch':best_epoch});del opt;return m,best_epoch

def probe():
 start_guard();pp=cfg();hw=gpu_setup();df,raw,y,splits,_,_,_=load_split('primary');ext,_=external('both',df.id);q=np.clip(ext.orig_teacher_lgb.to_numpy(),1e-6,1-1e-6);teacher=np.log(q/(1-q)).astype('float32');fi=splits[0][0];tr=FTTFeatures();x=tr.fit_transform(raw.iloc[fi],teacher[fi]);out=ROOT/'research_artifacts/w3_ftt_small_probe';out.mkdir(exist_ok=False);rec={'status':'running','started_utc':now(),'hardware':hw,'score_evaluated':False,'outer_validation_labels_used':False,'attempts':[]};dump(out/'result.json',rec)
 for bs in pp.get('probe_batch_rows',[256,128]):
  m=None;opt=None;torch.cuda.reset_peak_memory_stats()
  try:
   torch.manual_seed(42);m=create(x[0].shape[1],tr.cardinalities).to('cuda');opt=torch.optim.AdamW(m.make_parameter_groups(),lr=1e-4,weight_decay=1e-5,foreach=False);started=time.monotonic();m.train()
   for j in range(64):
    ix=np.arange(j*bs,(j+1)*bs);n,c=batch(x,ix);t=torch.from_numpy(y[fi[ix]].astype('float32')).to('cuda');opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.binary_cross_entropy_with_logits(m(n,c).flatten(),t)
    assert torch.isfinite(loss);loss.backward();opt.step()
   torch.cuda.synchronize();seconds=time.monotonic()-started;pv=predict(m,(x[0][:256],x[1][:256]));assert np.isfinite(pv).all();used=own_gpu_mib();estimate=seconds/64*np.ceil(len(fi)/bs)*12*3.8;chosen={'batch_rows':bs,'64_steps_seconds':seconds,'worst_primary_train_seconds':estimate,'process_gpu_mib':used,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'status':'measured'};rec['attempts'].append(chosen);dump(out/'result.json',rec);assert used is None or used<=1024,'Process GPU exceeds1GiB';assert estimate<=1200,'Projected primary runtime exceeds20min';chosen['status']='complete';rec.update(status='complete',chosen=chosen,completed_utc=now());dump(out/'result.json',rec);dump(ROOT/'research_w3_ftt_hardware.json',rec);return
  except torch.OutOfMemoryError as e:rec['attempts'].append({'batch_rows':bs,'status':'oom','error':str(e),'ended_utc':now()});dump(out/'result.json',rec)
  finally:
   del m,opt;gc.collect();torch.cuda.empty_cache()
 raise RuntimeError('No FT-Transformer batch satisfies VRAM guard')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);a=ap.parse_args();start_guard()
 if a.split=='probe':probe();return
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_ftt_{prev}/result.json').read_text())['eligible_for_next']
 pp=cfg();assert json.loads((ROOT/'research_w3_ftt_hardware.json').read_text())['chosen']['batch_rows']==pp['chosen_batch_rows'];hw=gpu_setup();df,raw,y,splits,ref,_,fold=load_split(a.split);ext,_=external('both',df.id);q=np.clip(ext.orig_teacher_lgb.to_numpy(),1e-6,1-1e-6);teacher=np.log(q/(1-q)).astype('float32');out=ROOT/f'research_artifacts/w3_ftt_{a.split}';out.mkdir(exist_ok=True);allpred=np.full(len(y),np.nan);comps=[]
 with threadpool_limits(6):
  for k,(fi,vi) in enumerate(splits):
   start_guard();run=out/f'fold{k}';run.mkdir(exist_ok=True);rec={'status':'inner_selection','started_utc':now(),'plan_sha256':sha(ROOT/'research_w3_ftt_plan.json'),'runner_sha256':sha(__file__),'hardware':hw,'train_rows':len(fi),'valid_rows':len(vi),'outer_validation_used_in_fit':False};dump(run/'result.json',rec);ia,ib=train_test_split(np.arange(len(fi)),test_size=.1,stratify=y[fi],random_state=20261018);np.savez_compressed(run/'inner_split.npz',train_idx=fi[ia],valid_idx=fi[ib]);it=FTTFeatures();ix=it.fit_transform(raw.iloc[fi[ia]],teacher[fi[ia]]);iv=it.transform(raw.iloc[fi[ib]],teacher[fi[ib]]);joblib.dump(it,run/'inner_transformer.joblib');tr=FTTFeatures();x=tr.fit_transform(raw.iloc[fi],teacher[fi]);v=tr.transform(raw.iloc[vi],teacher[vi]);joblib.dump(tr,run/'transformer.joblib');predictions=[];selected=[]
   for j,seed in enumerate(pp['seeds']):
    member=run/f'member{j}';member.mkdir(exist_ok=True);model,ne=train(ix,y[fi[ia]],it.cardinalities,seed,member/'inner',pp['max_epochs'],(iv,y[fi[ib]]));del model;gc.collect();torch.cuda.empty_cache();selected.append(ne);start_guard();rec.update(status='outer_refit',current_member=j,selected_epochs=selected);dump(run/'result.json',rec);model,_=train(x,y[fi],tr.cardinalities,seed,member/'outer',ne);pred=predict(model,v);del model;gc.collect();torch.cuda.empty_cache();loaded=load_model(member/'outer/model.pt');again=predict(loaded,v);diff=float(np.max(np.abs(pred-again)));assert diff<1e-7;del loaded;gc.collect();torch.cuda.empty_cache();np.savez_compressed(member/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),pred=pred);dump(member/'result.json',{'status':'complete','selected_epochs':ne,'model_reload_max_abs_diff':diff,'model_sha256':sha(member/'outer/model.pt'),'completed_utc':now()});predictions.append(pred)
   pred=np.mean(predictions,axis=0);r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);allpred[vi]=pred;comps.append(c);np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r);rec.update(status='complete',completed_utc=now(),vs_v6=c,selected_epochs=selected);dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps,'completed_utc':now()});return
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,**({'fold':fold} if fold is not None else {}));res={'status':'complete','split':a.split,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_w3_ftt_last_failure.json',{'status':'failed','ended_utc':now(),'argv':os.sys.argv,'error':repr(e)});raise

"""Frozen four-member DeepFM, training-only preprocessing, capped GPU and no early stopping."""
from research_w3_common import *
import argparse,time,gc,math,joblib,subprocess
from torch import nn
from threadpoolctl import threadpool_limits
from research_w3_prior_features import PriorFeatures
from research_external_ablation import external
def own_gpu_mib():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  a=line.split(',')
  if len(a)==2 and a[0].strip()==str(os.getpid()):
   try:return int(a[1].strip())
   except ValueError:pass
 return None
class DeepFM(nn.Module):
 def __init__(self,sizes,n_num,arch):
  super().__init__();self.register_buffer('offsets',torch.tensor(np.r_[0,np.cumsum(sizes)[:-1]],dtype=torch.long));n=int(sum(sizes));d=arch['embedding_dim'];self.emb=nn.Embedding(n,d);self.linear=nn.Embedding(n,2);nn.init.normal_(self.emb.weight,std=arch['embedding_init_std']);nn.init.zeros_(self.linear.weight)
  with torch.no_grad():self.emb.weight[self.offsets]=0
  self.num_bn=nn.BatchNorm1d(n_num);layers=[];width=len(sizes)*d+n_num
  for h in arch['hidden']:layers.extend([nn.Linear(width,h),nn.BatchNorm1d(h),nn.SiLU(),nn.Dropout(arch['dropout'])]);width=h
  layers.append(nn.Linear(width,2));self.mlp=nn.Sequential(*layers);self.bias=nn.Parameter(torch.zeros(2));self.fm_scales=nn.Parameter(torch.zeros(2));self.max_rate=arch['max_noise_rate'];self.register_buffer('base_rate_logits',torch.logit(torch.as_tensor(arch['base_rates'],dtype=torch.float32)/self.max_rate));nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
 def forward(self,n,c,prior):
  idx=c+self.offsets;e=self.emb(idx);pair=.5*(e.sum(1).square()-e.square().sum(1)).sum(1);heads=self.bias+self.linear(idx).sum(1)+pair[:,None]*self.fm_scales+self.mlp(torch.cat([e.flatten(1),self.num_bn(n)],1));rates=self.max_rate*torch.sigmoid(heads+self.base_rate_logits);q=torch.sigmoid(prior);prob=rates[:,0]+(1-rates.sum(1))*q;return torch.logit(prob.clamp(1e-6,1-1e-6))
def predict(m,arrays,device='cuda'):
 m.eval();out=[]
 with torch.inference_mode():
  for j in range(0,len(arrays[0]),2048):out.append(m(torch.from_numpy(arrays[0][j:j+2048]).to(device),torch.from_numpy(arrays[1][j:j+2048]).to(device),torch.from_numpy(arrays[2][j:j+2048]).to(device)).sigmoid().cpu().numpy())
 return np.concatenate(out)
def load(path,device='cuda'):
 z=torch.load(path,map_location='cpu',weights_only=True);m=DeepFM(z['sizes'],z['n_num'],z['architecture']);m.load_state_dict(z['state_dict']);return m.to(device)
def fit_member(X,y,pp,seed,out,rec,probe=False):
 torch.manual_seed(seed);np.random.seed(seed);cfg=pp['training'];n=torch.from_numpy(X[0]).to('cuda');c=torch.from_numpy(X[1]).to('cuda');target=torch.as_tensor(y,dtype=torch.float32,device='cuda');prior=torch.from_numpy(X[2]).to('cuda');m=DeepFM(rec['cat_cardinalities'],n.shape[1],pp['architecture']).to('cuda');opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);epochs=1 if probe else cfg['epochs'];steps=math.ceil(len(y)/cfg['batch_size']);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg['lr'],total_steps=epochs*steps,pct_start=.2);rec['epochs']=[];start=time.monotonic()
 for epoch in range(epochs):
  t=time.monotonic();m.train();total=0.;count=0
  for ix in torch.randperm(len(y),device='cuda').split(cfg['batch_size']):
   # All current datasets have a final batch>1, required by batch normalization.
   assert len(ix)>1;opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(m(n[ix],c[ix],prior[ix]),target[ix]);assert torch.isfinite(loss);loss.backward();opt.step();sched.step();total+=float(loss.detach())*len(ix);count+=len(ix)
  torch.cuda.synchronize();used=own_gpu_mib();assert used is None or used<=1024,'ProcessGPU exceeds1GiB';row={'epoch':epoch+1,'seconds':time.monotonic()-t,'train_loss':total/count,'gpu_free_bytes':gpu_epoch_guard(),'process_gpu_mib':used};rec['epochs'].append(row);dump(out/'result.json',rec);print(out,row,flush=True)
 rec.update(fit_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());del n,c,target,prior,opt,sched
 if not probe:torch.save({'sizes':rec['cat_cardinalities'],'n_num':X[0].shape[1],'architecture':pp['architecture'],'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()}},out/'model.pt')
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);ap.add_argument('--batch-size',type=int,choices=[1024,2048],default=2048);a=ap.parse_args();start_guard();
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_noise_factor_{prev}_b{a.batch_size}/result.json').read_text())['eligible_for_next']
 hw=setup_gpu();torch.cuda.set_per_process_memory_fraction(778043392/torch.cuda.get_device_properties(0).total_memory);pp=json.loads((ROOT/'research_w3_noise_factor_plan.json').read_text());pp['training']['batch_size']=a.batch_size;df,raw,y,splits,ref,old,fold=load_split('primary' if a.split=='probe' else a.split);teacher,_=external('both',df.id);teacher=teacher.orig_teacher_lgb.to_numpy();out=ROOT/'research_artifacts'/f'w3_noise_factor_{a.split}_b{a.batch_size}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);control_all=np.full(len(y),np.nan);comparisons=[]
 for k,(fi,vi) in enumerate(splits):
  low=teacher[fi]<.01;high=teacher[fi]>.99;assert low.sum()>1000 and high.sum()>1000;rates=np.clip([(y[fi][low].sum()+1)/(low.sum()+2),((1-y[fi][high]).sum()+1)/(high.sum()+2)],1e-4,.2499);pp['architecture']['base_rates']=rates.tolist();
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'running','started_utc':now(),'hardware':hw,'plan_sha256':sha(ROOT/'research_w3_noise_factor_plan.json'),'runner_sha256':sha(__file__),'feature_code_sha256':sha(ROOT/'research_w3_prior_features.py'),'source_teacher_model_sha256':sha(ROOT/'research_artifacts/original_teacher_v1/model.txt'),'train_rows':len(fi),'valid_rows':len(vi),'validation_used_in_fit':False,'initial_rates':rates.tolist(),'anchor_counts':[int(low.sum()),int(high.sum())]};dump(run/'result.json',rec)
  try:
   with threadpool_limits(6):
    tr=PriorFeatures();X=tr.fit_transform(raw.iloc[fi],teacher[fi]);V=None if a.split=='probe' else tr.transform(raw.iloc[vi],teacher[vi]);joblib.dump(tr,run/'transformer.joblib');rec['cat_cardinalities']=tr.cardinalities;members=[];recs=[]
    for i,seed in enumerate(pp['training']['seeds'][:1] if a.split=='probe' else pp['training']['seeds']):
     start_guard();member=run/f'member{i}';member.mkdir();mr={'status':'running','seed':seed,'cat_cardinalities':tr.cardinalities};m=fit_member(X,y[fi],pp,seed,member,mr,probe=a.split=='probe')
     if a.split=='probe':
      pred=predict(m,tuple(a[:1024] for a in X));assert np.isfinite(pred).all();mr.update(status='complete',score_evaluated=False,completed_utc=now());dump(member/'result.json',mr);dump(out/'result.json',mr);return
     pred=predict(m,V);m.to('cpu');del m;gc.collect();torch.cuda.empty_cache();m=load(member/'model.pt');again=predict(m,V);diff=float(np.max(np.abs(pred-again)));assert diff<1e-7;mr.update(status='complete',model_reload_max_abs_diff=diff,model_sha256=sha(member/'model.pt'),completed_utc=now());dump(member/'result.json',mr);recs.append(mr);members.append(pred);del m;gc.collect();torch.cuda.empty_cache()
   pred=np.mean(np.asarray(members,dtype='float64'),axis=0);r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;constant=rates[0]+(1-rates.sum())*np.clip(teacher[vi],1e-6,1-1e-6);control_all[vi]=constant;control=.75*r+.25*constant;matched=compare(y[vi],control,candidate);c=compare(y[vi],r,candidate);comparisons.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,members=np.asarray(members));rec.update(status='complete',completed_utc=now(),vs_v6=c,vs_constant_rate_control=matched,members=recs);dump(run/'result.json',rec)
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comparisons});return
  except BaseException as e:rec.update(status='failed',error=repr(e),hardware_oom=isinstance(e,torch.OutOfMemoryError),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;control=.75*ref+.25*control_all[vi];matched=compare(y[vi],control,candidate);c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,**({'fold':fold} if fold is not None else {}));res={'status':'complete','variant':'noise_factor','split':a.split,'vs_v6':c,'vs_constant_rate_control':matched,'fold_comparisons':comparisons,'eligible_for_next':gate(c,a.split) and matched['delta_auc']>0 and all(x['delta_auc']>0 for x in comparisons),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

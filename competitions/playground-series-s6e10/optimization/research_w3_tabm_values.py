"""Fixed-epoch TabM from the installed PyTabKit core; no validation-driven fitting."""
from research_w3_common import *
import argparse,time,gc,joblib,copy
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from pytabkit.models.nn_models.tabm import Model,make_parameter_groups
from pytabkit.models.nn_models.rtdl_num_embeddings import compute_bins
from research_w3_tabm_values_features import TabMValueFeatures as TabMFeatures
def predict(model,arrays):
 model.eval();out=[]
 with torch.inference_mode():
  for j in range(0,len(arrays[0]),512):
   n=torch.from_numpy(arrays[0][j:j+512]).to('cuda');c=torch.from_numpy(arrays[1][j:j+512]).to('cuda');out.append(model(n,c).softmax(-1).mean(1)[:,1].cpu().numpy())
 return np.concatenate(out)
def fit(X,y,cfg,run,rec,probe=False):
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);n=torch.from_numpy(X[0]).to('cuda');c=torch.from_numpy(X[1]).to('cuda');target=torch.as_tensor(y,dtype=torch.long,device='cuda');bins=compute_bins(n,n_bins=cfg['num_bins']);args={'n_num_features':n.shape[1],'cat_cardinalities':rec['cat_cardinalities'],'n_classes':2,'backbone':{'type':'MLP','n_blocks':cfg['n_blocks'],'d_block':cfg['d_block'],'dropout':cfg['dropout']},'bins':bins,'num_embeddings':{'type':'PiecewiseLinearEmbeddings','d_embedding':cfg['d_embedding'],'activation':False,'version':'B'},'arch_type':'tabm','k':cfg['k'],'share_training_batches':True};m=Model(**args).to('cuda');opt=torch.optim.AdamW(make_parameter_groups(m),lr=cfg['lr'],weight_decay=cfg['weight_decay']);rec['epochs']=[];batch=cfg['batch_size'];epochs=1 if probe else cfg['epochs'];start=time.monotonic()
 for epoch in range(epochs):
  t=time.monotonic();m.train();loss_sum=0.;count=0
  for idx in torch.randperm(len(y),device='cuda').split(batch):
   opt.zero_grad(set_to_none=True);logits=m(n[idx],c[idx]);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target[idx].repeat_interleave(cfg['k']));assert torch.isfinite(loss);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(idx);count+=len(idx)
  torch.cuda.synchronize();row={'epoch':epoch+1,'seconds':time.monotonic()-t,'train_loss':loss_sum/count,'gpu_free_bytes':gpu_epoch_guard()};rec['epochs'].append(row);dump(run/'result.json',rec);print(run.parent.name,run.name,row,flush=True)
 rec.update(fit_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());del n,c,target,opt
 if not probe:
  state={k:v.detach().cpu() for k,v in m.state_dict().items()};savedargs=copy.deepcopy(args);savedargs['bins']=[t.cpu() for t in args['bins']];torch.save({'args':savedargs,'state_dict':state},run/'model.pt')
 return m
def load(path,device='cuda'):
 ck=torch.load(path,map_location='cpu',weights_only=True);m=Model(**ck['args']);m.load_state_dict(ck['state_dict']);return m.to(device)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['probe','primary','confirm','oof'],required=True);ap.add_argument('--batch-size',type=int,choices=[128,256],default=256);a=ap.parse_args();start_guard();hw=setup_gpu();cfg=dict(plan()['tabm_config'],batch_size=a.batch_size);df,raw,y,splits,ref,old,fold=load_split('primary' if a.split=='probe' else a.split);out=ROOT/'research_artifacts'/f'w3_tabm_values_{a.split}_b{a.batch_size}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);comparisons=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'running','started_utc':now(),'hardware':hw,'config':cfg,'plan_sha256':sha(ROOT/'research_w3_plan.json'),'feature_expansion_plan_sha256':sha(ROOT/'research_w3_tabm_values_plan.json'),'feature_code_sha256':sha(ROOT/'research_w3_tabm_values_features.py'),'runner_sha256':sha(__file__),'train_rows':len(fi),'valid_rows':len(vi),'validation_used_in_fit':False};dump(run/'result.json',rec)
  try:
   with threadpool_limits(6):
    tr=TabMFeatures();X=tr.fit_transform(raw.iloc[fi]);V=None if a.split=='probe' else tr.transform(raw.iloc[vi]);joblib.dump(tr,run/'transformer.joblib');rec['cat_cardinalities']=tr.cardinalities;m=fit(X,y[fi],cfg,run,rec,probe=a.split=='probe')
    if a.split=='probe':
     pred=predict(m,(X[0][:1024],X[1][:1024]));assert np.isfinite(pred).all();rec.update(status='complete',score_evaluated=False,completed_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);return
    pred=predict(m,V);m.to('cpu');del m;gc.collect();torch.cuda.empty_cache();m=load(run/'model.pt');again=predict(m,V);diff=float(np.max(np.abs(pred-again)));assert diff<1e-7;del m;gc.collect();torch.cuda.empty_cache()
   r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);comparisons.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate);rec.update(status='complete',completed_utc=now(),auc=float(roc_auc_score(y[vi],pred)),vs_v6=c,model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.pt'));dump(run/'result.json',rec)
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comparisons});return
  except BaseException as e:rec.update(status='failed',error=repr(e),hardware_oom=isinstance(e,torch.OutOfMemoryError),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,**({'fold':fold} if fold is not None else {}));res={'status':'complete','variant':'tabm_values','split':a.split,'vs_v6':c,'fold_comparisons':comparisons,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comparisons),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

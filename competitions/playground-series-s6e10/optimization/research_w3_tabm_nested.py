"""TabM epoch selection strictly inside training; fresh refit never sees outer validation."""
from research_w3_common import *
import argparse,time,gc,joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from pytabkit.models.nn_models.tabm import Model,make_parameter_groups
from pytabkit.models.nn_models.rtdl_num_embeddings import compute_bins
from research_w3_tabm_values import fit,predict,load
from research_w3_tabm_values_features import TabMValueFeatures
def select_epochs(raw,y,global_idx,cfg,out):
 # This function accepts training-only data and cannot access any outer holdout labels.
 pp=json.loads((ROOT/'research_w3_tabm_nested_plan.json').read_text())['inner_selection'];a,b=train_test_split(np.arange(len(y)),test_size=pp['validation_fraction'],stratify=y,random_state=pp['split_seed']);np.savez_compressed(out/'inner_split.npz',train_idx=global_idx[a],valid_idx=global_idx[b]);tr=TabMValueFeatures();X=tr.fit_transform(raw.iloc[a]);V=tr.transform(raw.iloc[b]);joblib.dump(tr,out/'inner_transformer.joblib');torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);n=torch.from_numpy(X[0]).to('cuda');c=torch.from_numpy(X[1]).to('cuda');target=torch.as_tensor(y[a],dtype=torch.long,device='cuda');bins=compute_bins(n,n_bins=cfg['num_bins']);m=Model(n_num_features=n.shape[1],cat_cardinalities=tr.cardinalities,n_classes=2,backbone={'type':'MLP','n_blocks':cfg['n_blocks'],'d_block':cfg['d_block'],'dropout':cfg['dropout']},bins=bins,num_embeddings={'type':'PiecewiseLinearEmbeddings','d_embedding':cfg['d_embedding'],'activation':False,'version':'B'},arch_type='tabm',k=cfg['k'],share_training_batches=True).to('cuda');opt=torch.optim.AdamW(make_parameter_groups(m),lr=cfg['lr'],weight_decay=cfg['weight_decay']);rec={'status':'running','inner_train_rows':len(a),'inner_valid_rows':len(b),'outer_validation_used':False,'curve':[]};best=-np.inf;bestepoch=0;bad=0
 for epoch in range(1,pp['max_epochs']+1):
  t=time.monotonic();m.train();loss_sum=0.;count=0
  for ix in torch.randperm(len(a),device='cuda').split(cfg['batch_size']):
   opt.zero_grad(set_to_none=True);logits=m(n[ix],c[ix]);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target[ix].repeat_interleave(cfg['k']));assert torch.isfinite(loss);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix);count+=len(ix)
  pred=predict(m,V);auc=float(roc_auc_score(y[b],pred));torch.cuda.synchronize();row={'epoch':epoch,'seconds':time.monotonic()-t,'train_loss':loss_sum/count,'inner_validation_auc':auc,'gpu_free_bytes':gpu_epoch_guard()};rec['curve'].append(row)
  if auc>best:best=auc;bestepoch=epoch;bad=0
  else:bad+=1
  rec.update(best_epoch=bestepoch,best_inner_auc=best,consecutive_nonimproving=bad);dump(out/'inner_selection.json',rec);print('inner',out.parent.name,out.name,row,flush=True)
  if bad>=pp['patience']:break
 rec.update(status='complete',completed_utc=now(),inner_preprocessing_train_only=True,inner_model_discarded=True);dump(out/'inner_selection.json',rec);del m,n,c,target,opt;gc.collect();torch.cuda.empty_cache();return bestepoch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();hw=setup_gpu();cfg=dict(plan()['tabm_config']);df,raw,y,splits,ref,old,fold=load_split(a.split);out=ROOT/'research_artifacts'/f'w3_tabm_nested_{a.split}';out.mkdir(exist_ok=False);allpred=np.full(len(y),np.nan);comparisons=[];epochs=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir();rec={'status':'inner_selection','started_utc':now(),'hardware':hw,'plan_sha256':sha(ROOT/'research_w3_tabm_nested_plan.json'),'runner_sha256':sha(__file__),'feature_code_sha256':sha(ROOT/'research_w3_tabm_values_features.py'),'train_rows':len(fi),'valid_rows':len(vi),'outer_validation_used_in_fit':False};dump(run/'result.json',rec)
  try:
   with threadpool_limits(6):
    ne=select_epochs(raw.iloc[fi].reset_index(drop=True),y[fi],fi,cfg,run);start_guard();localcfg=dict(cfg,epochs=ne);epochs.append(ne);tr=TabMValueFeatures();X=tr.fit_transform(raw.iloc[fi]);V=tr.transform(raw.iloc[vi]);joblib.dump(tr,run/'transformer.joblib');rec.update(status='full_outer_training',cat_cardinalities=tr.cardinalities,config=localcfg,selected_epochs=ne);m=fit(X,y[fi],localcfg,run,rec);pred=predict(m,V);m.to('cpu');del m;gc.collect();torch.cuda.empty_cache();m=load(run/'model.pt');again=predict(m,V);diff=float(np.max(np.abs(pred-again)));assert diff<1e-7;del m;gc.collect();torch.cuda.empty_cache()
   r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);comparisons.append(c);allpred[vi]=pred;np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate);rec.update(status='complete',completed_utc=now(),raw_auc=float(roc_auc_score(y[vi],pred)),vs_v6=c,model_reload_max_abs_diff=diff,model_sha256=sha(run/'model.pt'));dump(run/'result.json',rec)
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comparisons,'selected_epochs':epochs});return
  except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(run/'result.json',rec);dump(out/'result.json',rec);raise
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,**({'fold':fold} if fold is not None else {}));res={'status':'complete','variant':'tabm_nested','split':a.split,'vs_v6':c,'selected_epochs':epochs,'fold_comparisons':comparisons,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comparisons),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

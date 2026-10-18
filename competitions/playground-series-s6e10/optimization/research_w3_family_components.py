"""Fit unchanged component recipes only for the gated, frozen family ensemble."""
from pathlib import Path
import os
os.environ['CUDA_CACHE_PATH']=str(Path(__file__).absolute().parent.parent/'.cache-family-w3/cuda')
from research_w3_common import *
import argparse,time,gc,joblib,importlib
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split
from research_external_ablation import external
NAMES={'tabm':'w3_tabm_{split}_b256','tabm_nested':'w3_tabm_nested_{split}','deepfm':'w3_deepfm_{split}_b2048','rank_deepfm':'w3_rank_deepfm_{split}_b2048','prior_deepfm':'w3_prior_deepfm_{split}_b2048','ftt':'w3_ftt_{split}'}
def valid_file(path,vi,df,y):
 z=np.load(path);assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);assert np.isfinite(z['pred']).all();return z['pred'].astype('float64')
def merge(component,split,df,y,splits,fold):
 out=ROOT/'research_artifacts'/NAMES[component].format(split=split);allpred=np.full(len(y),np.nan)
 if (out/'validation.npz').exists() and (out/'result.json').exists():
  expected=np.arange(len(y)) if split=='oof' else splits[0][1];valid_file(out/'validation.npz',expected,df,y);assert json.loads((out/'result.json').read_text())['status']=='complete';return True
 for k,(_,vi) in enumerate(splits):
  p=out/f'fold{k}/validation.npz'
  if not p.exists():return False
  allpred[vi]=valid_file(p,vi,df,y)
 vi=np.arange(len(y)) if split=='oof' else splits[0][1];assert np.isfinite(allpred[vi]).all();np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=allpred[vi],**({'fold':fold} if fold is not None else {}));dump(out/'result.json',{'status':'complete','fit_purpose':'frozen family ensemble component only','eligible_for_individual_promotion':False,'family_plan_sha256':sha(ROOT/'research_w3_family_fusion_plan.json'),'prediction_sha256':sha(out/'validation.npz'),'completed_utc':now()});return True
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--component',choices=list(NAMES),required=True);ap.add_argument('--split',choices=['confirm','oof'],required=True);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args();start_guard()
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm']):assert json.loads((ROOT/f'research_artifacts/w3_family_fusion_{prev}/result.json').read_text())['eligible_for_next']
 df,raw,y,splits,_,_,fold=load_split(a.split);fi,vi=splits[a.fold];out=ROOT/'research_artifacts'/NAMES[a.component].format(split=a.split);out.mkdir(exist_ok=True);run=out/f'fold{a.fold}'
 if (run/'validation.npz').exists():
  assert json.loads((run/'result.json').read_text())['status']=='complete';valid_file(run/'validation.npz',vi,df,y);merge(a.component,a.split,df,y,splits,fold);print('reused',str(run),flush=True);return
 run.mkdir(exist_ok=False);hw=setup_gpu();torch.cuda.set_per_process_memory_fraction(778043392/torch.cuda.get_device_properties(0).total_memory);ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();rec={'status':'training','started_utc':now(),'component':a.component,'split':a.split,'fold':a.fold,'fit_purpose':'frozen family ensemble only; no standalone promotion','family_plan_sha256':sha(ROOT/'research_w3_family_fusion_plan.json'),'runner_sha256':sha(__file__),'train_rows':len(fi),'valid_rows':len(vi),'outer_validation_used_in_fit':False,'hardware':hw};dump(run/'result.json',rec);started=time.monotonic()
 try:
  with threadpool_limits(6):
   lib=importlib.import_module('research_w3_'+a.component);rec['unchanged_recipe_module_sha256']=sha(lib.__file__)
   if a.component in ['tabm','tabm_nested']:
    if a.component=='tabm':tr=lib.TabMFeatures();cc=dict(plan()['tabm_config'],batch_size=256)
    else:
     from research_w3_tabm_values import fit,predict,load
     cc=dict(plan()['tabm_config']);ne=lib.select_epochs(raw.iloc[fi].reset_index(drop=True),y[fi],fi,cc,run);cc['epochs']=ne;rec['selected_epochs']=ne;tr=lib.TabMValueFeatures()
    x=tr.fit_transform(raw.iloc[fi]);v=tr.transform(raw.iloc[vi]);joblib.dump(tr,run/'transformer.joblib');rec['cat_cardinalities']=tr.cardinalities
    if a.component=='tabm':model=lib.fit(x,y[fi],cc,run,rec);pred=lib.predict(model,v);loader=lib.load;predictor=lib.predict
    else:model=fit(x,y[fi],cc,run,rec);pred=predict(model,v);loader=load;predictor=predict
    del model;gc.collect();torch.cuda.empty_cache();model=loader(run/'model.pt');again=predictor(model,v);rd=float(np.max(np.abs(pred-again)));assert rd<1e-7;rec['model_reload_max_abs_diff']=rd;rec['model_sha256']=sha(run/'model.pt');del model;gc.collect();torch.cuda.empty_cache()
   elif a.component in ['deepfm','rank_deepfm','prior_deepfm']:
    pp=json.loads((ROOT/f'research_w3_{a.component}_plan.json').read_text());assert pp['training']['batch_size']==2048
    if a.component=='prior_deepfm':tr=lib.PriorFeatures();x=tr.fit_transform(raw.iloc[fi],teacher[fi]);v=tr.transform(raw.iloc[vi],teacher[vi])
    else:tr=lib.TabMValueFeatures();x=tr.fit_transform(raw.iloc[fi]);v=tr.transform(raw.iloc[vi])
    joblib.dump(tr,run/'transformer.joblib');predictions=[];members=[]
    for j,seed in enumerate(pp['training']['seeds']):
     start_guard();member=run/f'member{j}';member.mkdir();mr={'status':'training','seed':seed,'cat_cardinalities':tr.cardinalities};model=lib.fit_member(x,y[fi],pp,seed,member,mr);pv=lib.predict(model,v);del model;gc.collect();torch.cuda.empty_cache();loaded=lib.load(member/'model.pt');again=lib.predict(loaded,v);rd=float(np.max(np.abs(pv-again)));assert rd<1e-7;mr.update(status='complete',model_reload_max_abs_diff=rd,model_sha256=sha(member/'model.pt'),completed_utc=now());dump(member/'result.json',mr);members.append(mr);predictions.append(pv);del loaded;gc.collect();torch.cuda.empty_cache()
    pred=np.mean(np.asarray(predictions,dtype='float64'),axis=0);rec['members']=members
   else:
    pp=lib.cfg();p=np.clip(teacher,1e-6,1-1e-6);logit=np.log(p/(1-p)).astype('float32');ia,ib=train_test_split(np.arange(len(fi)),test_size=.1,stratify=y[fi],random_state=20261018);np.savez_compressed(run/'inner_split.npz',train_idx=fi[ia],valid_idx=fi[ib]);it=lib.FTTFeatures();ix=it.fit_transform(raw.iloc[fi[ia]],logit[fi[ia]]);iv=it.transform(raw.iloc[fi[ib]],logit[fi[ib]]);joblib.dump(it,run/'inner_transformer.joblib');tr=lib.FTTFeatures();x=tr.fit_transform(raw.iloc[fi],logit[fi]);v=tr.transform(raw.iloc[vi],logit[vi]);joblib.dump(tr,run/'transformer.joblib');predictions=[];members=[]
    for j,seed in enumerate(pp['seeds']):
     member=run/f'member{j}';member.mkdir(exist_ok=True);model,ne=lib.train(ix,y[fi[ia]],it.cardinalities,seed,member/'inner',pp['max_epochs'],(iv,y[fi[ib]]));del model;gc.collect();torch.cuda.empty_cache();start_guard();model,_=lib.train(x,y[fi],tr.cardinalities,seed,member/'outer',ne);pv=lib.predict(model,v);del model;gc.collect();torch.cuda.empty_cache();loaded=lib.load_model(member/'outer/model.pt');again=lib.predict(loaded,v);rd=float(np.max(np.abs(pv-again)));assert rd<1e-7;mr={'status':'complete','selected_epochs':ne,'model_reload_max_abs_diff':rd,'model_sha256':sha(member/'outer/model.pt'),'completed_utc':now()};dump(member/'result.json',mr);members.append(mr);predictions.append(pv);del loaded;gc.collect();torch.cuda.empty_cache()
    pred=np.mean(predictions,axis=0);rec['members']=members
  assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred);rec.update(status='complete',completed_utc=now(),elapsed_seconds=time.monotonic()-started,prediction_sha256=sha(run/'validation.npz'),transformer_sha256=sha(run/'transformer.joblib'));dump(run/'result.json',rec);merge(a.component,a.split,df,y,splits,fold);print(json.dumps(rec,indent=2),flush=True)
 except BaseException as e:rec.update(status='failed',ended_utc=now(),error=repr(e));dump(run/'result.json',rec);raise
if __name__=='__main__':main()

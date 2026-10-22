"""Frozen2 domain learners, with separate fit/prediction receipts; no successful refits."""
from research_r93_repair_common import *
from research_r93_av import transformed,auc_report
import lightgbm as lgb,resource
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score

def main():
 p,repair=repaired_plan();assert json.loads((OUT/'basic.json').read_text())['status']=='complete_covariates_only';stop,thread=guard();tick=time.monotonic();ph=sha(ROOT/'research_r93_plan.json');rh=sha(ROOT/'research_r93_repair_manifest.json')
 try:
  if (OUT/'adversarial.json').exists():
   done=json.loads((OUT/'adversarial.json').read_text());assert done['status']=='complete_domain_diagnostic' and done['plan_sha256']==ph;print('REUSE_COMPLETE_ADVERSARIAL',flush=True);return
  tr,te,tri,tei=load();raw=pd.concat([tr,te],ignore_index=True);domain=np.r_[np.zeros(len(tr),'int8'),np.ones(len(te),'int8')];fi,vi,h=split(raw);assert (np.bincount(domain[vi],minlength=2)>=50000).all()
  X,voc=transformed(raw,fi);A=X.iloc[fi].reset_index(drop=True);V=X.iloc[vi].reset_index(drop=True);assert list(A)==list(V)==[f'f{i:03}' for i in range(21)]
  prep=dict(feature_names=list(X),raw_feature_mapping=dict(zip(list(X),FEATURES)),fit_category_vocabularies=voc,split_fit_rows=len(fi),split_holdout_rows=len(vi),fit_indices_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),holdout_indices_sha256=hashlib.sha256(vi.tobytes()).hexdigest(),exact_feature_hash_groups_disjoint=True,ID_or_row_index_as_feature=False,SAT_or_existing_model_scores_as_features=False)
  preppath=OUT/'preprocessing.json'
  if preppath.exists():assert json.loads(preppath.read_text())==prep
  else:reserve();dump(preppath,prep)
  perm=np.random.default_rng(20261023).permutation(domain[fi]);pred={};models=[]
  for name,yfit in [('real',domain[fi]),('permuted',perm)]:
   modelpath=OUT/f'domain_{name}.txt';fitpath=OUT/f'domain_{name}_fit.json';predictionpath=OUT/f'domain_{name}_predictions.npz';receiptpath=OUT/f'domain_{name}_complete.json';labelhash=hashlib.sha256(yfit.tobytes()).hexdigest()
   if fitpath.exists():
    fit=json.loads(fitpath.read_text());assert fit['status']=='fit_complete' and fit['plan_sha256']==ph and fit['repair_sha256']==rh and fit['parameters']==p['adversarial']['parameters'] and fit['fit_labels_sha256']==labelhash and fit['preprocessing_sha256']==sha(preppath) and fit['model_sha256']==sha(modelpath);booster=lgb.Booster(model_file=str(modelpath));print('REUSE_COMPLETED_DOMAIN_FIT',name,flush=True)
   else:
    assert not modelpath.exists(),'Partialmodelwithoutfitreceiptrequiresassessment';reserve(4);begin=time.monotonic()
    model=lgb.LGBMClassifier(**p['adversarial']['parameters'])
    with threadpool_limits(6):model.fit(A,yfit)
    assert np.array_equal(model.classes_,[0,1]) and model.feature_name_==list(A);booster=model.booster_;api_error=float(np.max(np.abs(model.predict_proba(A.iloc[:256])[:,1]-booster.predict(A.iloc[:256],num_threads=6))));assert api_error<1e-12;booster.save_model(str(modelpath));fit=dict(classifier_to_booster_api_error=api_error,status='fit_complete',name=name,completed_utc=now(),plan_sha256=ph,repair_sha256=rh,parameters=p['adversarial']['parameters'],actual_trees=booster.num_trees(),fit_seconds=time.monotonic()-begin,model_sha256=sha(modelpath),preprocessing_sha256=sha(preppath),fit_labels_sha256=labelhash,class_order=[0,1]);reserve();dump(fitpath,fit);print('SAVED_COMPLETED_DOMAIN_FIT',name,fit['fit_seconds'],flush=True)
   if receiptpath.exists():
    receipt=json.loads(receiptpath.read_text());assert receipt['plan_sha256']==ph and receipt['fit_receipt_sha256']==sha(fitpath) and receipt['predictions_sha256']==sha(predictionpath)
    z=np.load(predictionpath);assert np.array_equal(z['idx'],vi);q=z['probability'];print('REUSE_COMPLETED_DOMAIN_PREDICTIONS',name,flush=True)
   else:
    begin=time.monotonic()
    with threadpool_limits(6):q=booster.predict(V,num_threads=6);fitq=booster.predict(A,num_threads=6)
    restored=lgb.Booster(model_file=str(modelpath));error=float(np.max(np.abs(q[:2048]-restored.predict(V.iloc[:2048],num_threads=6))));assert error<1e-12
    reserve(8);np.savez_compressed(predictionpath,idx=vi,probability=q);receipt=dict(status='predictions_complete',name=name,completed_utc=now(),plan_sha256=ph,fit_receipt_sha256=sha(fitpath),predictions_sha256=sha(predictionpath),reload_error=error,fit_AUC_to_fitted_labels=float(roc_auc_score(yfit,fitq)),prediction_seconds=time.monotonic()-begin);reserve();dump(receiptpath,receipt);print('SAVED_COMPLETED_DOMAIN_PREDICTIONS',name,flush=True)
   pred[name]=q;models.append(dict(name=name,parameters=p['adversarial']['parameters'],actual_trees=fit['actual_trees'],seconds=fit['fit_seconds']+receipt['prediction_seconds'],model_sha256=fit['model_sha256'],reload_error=receipt['reload_error'],fit_AUC_to_fitted_labels=receipt['fit_AUC_to_fitted_labels']))
   if name=='real':real_model=booster
  # No heldout scores until both learners/prediction stages completed or reused.
  real=auc_report(domain[vi],pred['real']);null=auc_report(domain[vi],pred['permuted']);null_ok=bool(abs(null['AUC']-.5)<=.01);material=bool(null_ok and real['conditional_95pct_CI'][0]>.55);importance=[]
  if material:
   for j,(name,cols) in enumerate(p['adversarial']['group_permutation_only_if_material']['groups'].items()):
    altered=V.copy();order=np.random.default_rng(20261026+j).permutation(len(V));names=[f'f{FEATURES.index(c):03}' for c in cols]
    for c in names:altered[c]=V[c].iloc[order].reset_index(drop=True)
    q=real_model.predict(altered,num_threads=6);importance.append(dict(group=name,fields=cols,holdout_AUC=float(roc_auc_score(domain[vi],q)),AUC_drop=float(real['AUC']-roc_auc_score(domain[vi],q)),one_fixed_permutation_only=True))
  reserve(12);np.savez_compressed(OUT/'heldout_predictions.npz',idx=vi,domain_target=domain[vi],real_probability=pred['real'],permuted_probability=pred['permuted'])
  result=dict(status='complete_domain_diagnostic',completed_utc=now(),plan_sha256=ph,repair_sha256=rh,source_positive_class='officialtest=1,officialtrain=0',official_SAT_loaded=False,features=FEATURES,models=models,real_holdout=real,permuted_fit_label_negative_control=null,negative_control_passed=null_ok,material_generalizable_distinguishability=material,predeclared_material_line='real95lower>.55 andabs(nullAUC-.5)<=.01; practicalscreeningline,notuniversaldefinitionofshift',group_permutation_if_material=importance,holdout_class_counts=np.bincount(domain[vi],minlength=2).tolist(),fit_class_counts=np.bincount(domain[fi],minlength=2).tolist(),preprocessing_sha256=sha(preppath),predictions_sha256=sha(OUT/'heldout_predictions.npz'),new_domain_classifier_fits=2,new_SAT_classifier_fits=0,GPU_used=False,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,interpretation='MembershippredictabilitycanindicateP(X)difference;neitherprovesP(Y|X)shift norpredictsAUCbenefitofreweighting. WeakfiniteAVcannotproveIIDorabsentconceptshift. No scoreorientationflipping/retuning.')
  reserve();dump(OUT/'adversarial.json',result);print('DOMAIN_DIAGNOSTIC_COMPLETE',json.dumps(dict(real=real,negative_control=null,material=material)),flush=True)
 finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':main()

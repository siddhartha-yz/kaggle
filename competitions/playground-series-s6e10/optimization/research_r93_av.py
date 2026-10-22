"""Exactly2 fixed-budget domain fits: real and permuted-fit-label negative control."""
from research_r93_common import *
import lightgbm as lgb,resource
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from compare_validation import compare

def transformed(raw,fi):
 out=raw.copy();voc={}
 for c in CATS:
  voc[c]=sorted(raw[c].iloc[fi].unique());out[c]=raw[c].astype(pd.CategoricalDtype(voc[c]))
 out.columns=[f'f{i:03}' for i in range(len(FEATURES))]
 assert len(out.columns)==21 and all(n.startswith('f') for n in out)
 return out,voc

def auc_report(y,p):
 r=compare(y,np.full(len(y),.5),p)
 return dict(AUC=r['candidate_auc'],conditional_95pct_CI=[.5+x for x in r['conditional_95pct_ci']],SE=r['paired_se'],domain_logloss=r['candidate_logloss'])

def main():
 p=plan();assert json.loads((OUT/'basic.json').read_text())['status']=='complete_covariates_only';stop,thread=guard();tick=time.monotonic()
 try:
  tr,te,tri,tei=load();raw=pd.concat([tr,te],ignore_index=True);domain=np.r_[np.zeros(len(tr),'int8'),np.ones(len(te),'int8')];fi,vi,h=split(raw)
  assert (np.bincount(domain[vi],minlength=2)>=50000).all();X,voc=transformed(raw,fi);A=X.iloc[fi].reset_index(drop=True);V=X.iloc[vi].reset_index(drop=True);assert list(A)==list(V)==[f'f{i:03}' for i in range(21)]
  perm=np.random.default_rng(20261023).permutation(domain[fi]);pred={};models=[]
  # Both complete before assessing holdout distinguishability. No eval_set/earlystopping.
  for name,yfit in [('real',domain[fi]),('permuted',perm)]:
   reserve(4);begin=time.monotonic();model=lgb.LGBMClassifier(**p['adversarial']['parameters'])
   with threadpool_limits(6):model.fit(A,yfit);q=model.predict_proba(V)[:,1];fitq=model.predict_proba(A)[:,1]
   assert np.array_equal(model.classes_,[0,1]) and model.feature_name_==list(A);model.booster_.save_model(str(OUT/f'domain_{name}.txt'))
   restored=lgb.Booster(model_file=str(OUT/f'domain_{name}.txt'));replay=restored.predict(V.iloc[:2048],num_threads=6);error=float(np.max(np.abs(q[:2048]-replay)));assert error<1e-12
   pred[name]=q;models.append(dict(name=name,parameters=p['adversarial']['parameters'],actual_trees=model.booster_.num_trees(),seconds=time.monotonic()-begin,model_sha256=sha(OUT/f'domain_{name}.txt'),reload_error=error,fit_AUC_to_fitted_labels=float(roc_auc_score(yfit,fitq))))
   if name=='real':real_model=model
   print('DOMAIN_FIT_COMPLETE',name,models[-1]['seconds'],flush=True)
  real=auc_report(domain[vi],pred['real']);null=auc_report(domain[vi],pred['permuted']);null_ok=bool(abs(null['AUC']-.5)<=.01);material=bool(null_ok and real['conditional_95pct_CI'][0]>.55)
  importance=[]
  if material:
   groups={'demographic_travel':['Age',*CATS],'distance':['Flight Distance'],'ratings':RATINGS,'delays':['Departure Delay in Minutes','Arrival Delay in Minutes']}
   for j,(name,cols) in enumerate(groups.items()):
    altered=V.copy();order=np.random.default_rng(20261026+j).permutation(len(V));names=[f'f{FEATURES.index(c):03}' for c in cols]
    for c in names:altered[c]=V[c].iloc[order].reset_index(drop=True)
    q=real_model.predict_proba(altered)[:,1];importance.append(dict(group=name,fields=cols,holdout_AUC=float(roc_auc_score(domain[vi],q)),AUC_drop=float(real['AUC']-roc_auc_score(domain[vi],q)),one_fixed_permutation_only=True))
  reserve(12);np.savez_compressed(OUT/'heldout_predictions.npz',idx=vi,domain_target=domain[vi],real_probability=pred['real'],permuted_probability=pred['permuted'])
  prep=dict(feature_names=list(X),raw_feature_mapping=dict(zip(list(X),FEATURES)),fit_category_vocabularies=voc,split_fit_rows=len(fi),split_holdout_rows=len(vi),fit_indices_sha256=hashlib.sha256(fi.tobytes()).hexdigest(),holdout_indices_sha256=hashlib.sha256(vi.tobytes()).hexdigest(),exact_feature_hash_groups_disjoint=True,ID_or_row_index_as_feature=False,SAT_or_existing_model_scores_as_features=False)
  reserve();dump(OUT/'preprocessing.json',prep)
  result=dict(status='complete_domain_diagnostic',completed_utc=now(),plan_sha256=sha(ROOT/'research_r93_plan.json'),source_positive_class='officialtest=1,officialtrain=0',official_SAT_loaded=False,features=FEATURES,models=models,real_holdout=real,permuted_fit_label_negative_control=null,negative_control_passed=null_ok,material_generalizable_distinguishability=material,predeclared_material_line='real95lower>.55 andabs(nullAUC-.5)<=.01; practicalscreeningline,notuniversaldefinitionofshift',group_permutation_if_material=importance,holdout_class_counts=np.bincount(domain[vi],minlength=2).tolist(),fit_class_counts=np.bincount(domain[fi],minlength=2).tolist(),preprocessing_sha256=sha(OUT/'preprocessing.json'),predictions_sha256=sha(OUT/'heldout_predictions.npz'),new_domain_classifier_fits=2,new_SAT_classifier_fits=0,GPU_used=False,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,interpretation='MembershippredictabilitycanindicateP(X)difference;neitherprovesP(Y|X)shift norpredictsAUCbenefitofreweighting. WeakfiniteAVcannotproveIIDorabsentconceptshift. No scoreorientationflipping/retuning.')
  reserve();dump(OUT/'adversarial.json',result);print('DOMAIN_DIAGNOSTIC_COMPLETE',json.dumps(dict(real=real,negative_control=null,material=material)),flush=True)
 finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':main()

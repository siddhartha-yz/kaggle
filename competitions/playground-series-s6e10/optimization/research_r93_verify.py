"""Saved-domain-model/heldout-identity replay; never touches satisfaction labels."""
from research_r93_common import *
from research_r93_av import transformed,auc_report
import lightgbm as lgb

def main():
 p=plan(all_inputs=True);b=json.loads((OUT/'basic.json').read_text());r=json.loads((OUT/'adversarial.json').read_text());assert b['SAT_labels_loaded']==r['official_SAT_loaded']==False and r['new_domain_classifier_fits']==2 and r['new_SAT_classifier_fits']==0
 tr,te,tri,tei=load();raw=pd.concat([tr,te],ignore_index=True);fi,vi,h=split(raw);domain=np.r_[np.zeros(len(tr),'int8'),np.ones(len(te),'int8')];z=np.load(OUT/'heldout_predictions.npz');assert sha(OUT/'heldout_predictions.npz')==r['predictions_sha256'] and np.array_equal(z['idx'],vi) and np.array_equal(z['domain_target'],domain[vi])
 X,voc=transformed(raw,fi);prep=json.loads((OUT/'preprocessing.json').read_text());assert sha(OUT/'preprocessing.json')==r['preprocessing_sha256'] and prep['fit_category_vocabularies']==voc and prep['fit_indices_sha256']==hashlib.sha256(fi.tobytes()).hexdigest() and prep['holdout_indices_sha256']==hashlib.sha256(vi.tobytes()).hexdigest()
 errors=[]
 for row,name in zip(r['models'],['real','permuted']):
  assert row['name']==name and sha(OUT/f'domain_{name}.txt')==row['model_sha256'] and row['parameters']==p['adversarial']['parameters'];model=lgb.Booster(model_file=str(OUT/f'domain_{name}.txt'));q=model.predict(X.iloc[vi[:4096]],num_threads=6);err=float(np.max(np.abs(q-z[f'{name}_probability'][:4096])));assert err<1e-12;errors.append(err)
  report=auc_report(domain[vi],z[f'{name}_probability']);expected=r['real_holdout'] if name=='real' else r['permuted_fit_label_negative_control'];assert abs(report['AUC']-expected['AUC'])<1e-12
 assert not np.intersect1d(h[fi],h[vi]).size
 result=dict(status='passed_diagnostic_integrity',completed_utc=now(),plan_sha256=sha(ROOT/'research_r93_plan.json'),source_and_dataset_hashes_verified=True,feature_names_exact21=True,ID_and_SAT_excluded=True,group_split_and_holdout_rows_exact=True,saved_model_replay_max_errors=errors,new_fits=0,files_sha256={n:sha(OUT/n) for n in ['basic.json','adversarial.json','heldout_predictions.npz','preprocessing.json','domain_real.txt','domain_permuted.txt']})
 reserve();dump(OUT/'integrity.json',result);print('DIAGNOSTIC_INTEGRITY_PASSED',errors,flush=True)
if __name__=='__main__':main()

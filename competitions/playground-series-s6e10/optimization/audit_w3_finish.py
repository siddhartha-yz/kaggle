"""Read-only closeout: recompute all comparisons, source lineage, nested splits, CPU portability."""
import os
os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
from research_w3_common import *
import joblib,cloudpickle
from threadpoolctl import threadpool_limits
from audit_w3_tabm_nested_v7 import audit_inner
from research_w3_tabm_values import load as load_tabm
from research_original import safe_original
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);train=pd.read_csv(ROOT/'data/train.csv');raw=train.drop(columns=['id','satisfaction']);y=train.satisfaction.astype('int8').to_numpy();report={'started_utc':now(),'new_model_fits_performed':False,'comparisons':{},'nested_split_audits':{}}
 with threadpool_limits(1):
  routes=['w3_teacher_nn_primary','w3_tabm_primary_b256','w3_tabm_values_primary_b256','w3_deepfm_primary_b2048','w3_tabm_nested_primary','w3_tabm_nested_confirm']
  for name in routes:
   split='confirm' if name.endswith('confirm') else 'primary';p=ROOT/'research_artifacts'/name;z=np.load(p/'validation.npz');v=np.load(ROOT/'research_artifacts'/f'gpu_value_raw_{split}_n2/validation.npz');ix=np.load(ROOT/('iteration_artifacts/split_confirm20261017.npz' if split=='confirm' else 'artifacts/split_seed42.npz'));assert all(np.array_equal(z[k],v[k]) for k in ['idx','id','target']);assert np.array_equal(z['idx'],ix['valid']) and np.array_equal(z['target'],y[ix['valid']]);assert np.array_equal(z['candidate'],.75*v['candidate']+.25*z['pred']);c=compare(z['target'],v['candidate'],z['candidate']);r=json.loads((p/'result.json').read_text());assert c==r['vs_v6'];assert gate(c,split)==r['eligible_for_next'];report['comparisons'][name]=c
   if 'nested' in name:report['nested_split_audits'][split]=audit_inner(p/'fold0',ix['train'],raw)
   if name=='w3_tabm_primary_b256':
    tr=joblib.load(p/'fold0/transformer.joblib');assert tr.medians.equals(raw.iloc[ix['train']][tr.num].median())
    for c2 in tr.cat:assert tr.vocabs[c2]=={v:i+1 for i,v in enumerate(sorted(raw.iloc[ix['train']][c2].fillna('__MISSING__').astype(str).unique()))}
    report['raw_tabm_training_only_medians_and_four_vocabularies']=True
  source,_,_,cols,source_audit=safe_original();p=ROOT/'research_artifacts/w3_source_nn';r=json.loads((p/'result.json').read_text());assert r['status']=='complete' and not r['synthetic_targets_used_in_teacher'] and not r['validation_scores_used'] and r['model_sha256']==sha(p/'model.pkl');tr=joblib.load(p/'transformer.joblib');assert tr.medians.equals(source[tr.numeric].median()) and tr.variant=='raw';m=cloudpickle.loads((p/'model.pkl').read_bytes());assert np.array_equal(m.classes_,[0,1]) and all(v.device.type=='cpu' for v in m.cv_alg_interface_.model.parameters());enc=m.x_converter_.cat_tf.named_transformers_['categorical'];catcols=m.x_converter_.cat_tf.transformers_[0][2]
  for c,actual in zip(catcols,enc.categories_):
   if c.startswith('value_cat_'):orig=tr.numeric[int(c.rsplit('_',1)[1])];expected=np.sort(source[orig].fillna(tr.medians[orig]).astype('int64').astype(str).unique())
   else:expected=np.sort(source[c].astype(str).unique())
   assert np.array_equal(expected,actual)
  idx=np.random.default_rng(20261018).choice(len(train),1003,replace=False);features=np.load(p/'train_features.npz');assert np.array_equal(features['id'],train.id);pred=m.predict_proba(tr.transform(raw.iloc[idx],pd.DataFrame(index=range(len(idx)))))[:,1];diff=float(np.max(np.abs(pred-features['pred'][idx])));assert diff<2e-6;report['source_teacher']={'audit':source_audit,'all21_native_category_vocabularies_fit_only_safe_original':True,'source_only_medians':True,'cpu_inference_max_abs_diff':diff,'sample_rows':1003};del m,source
  p=ROOT/'research_artifacts/w3_tabm_nested_confirm/fold0';tr=joblib.load(p/'transformer.joblib');m=load_tabm(p/'model.pt','cpu');z=np.load(p/'validation.npz');j=np.random.default_rng(20261018).choice(len(z['idx']),1003,replace=False);arrays=tr.transform(raw.iloc[z['idx'][j]]);m.eval()
  with torch.inference_mode():pred=m(torch.from_numpy(arrays[0]),torch.from_numpy(arrays[1])).softmax(-1).mean(1)[:,1].numpy()
  diff=float(np.max(np.abs(pred-z['pred'][j])));assert diff<2e-6;report['nested_tabm_cpu_portability']={'sample_rows':1003,'max_abs_diff':diff,'all_tensors_cpu':all(v.device.type=='cpu' for v in m.state_dict().values())}
  expected='36ea4ef80e1dd1911257f240391d9cdb8474b8b5deb5c52733582da80bdd1aaa';assert sha(ROOT/'submission_realmlp_value_v6.csv')==expected;report['v6_unchanged_sha256']=expected;assert not list(ROOT.glob('submission_*v7.csv'));assert not list((ROOT/'research_artifacts').glob('w3*_oof*'));assert not list((ROOT/'research_artifacts').glob('w3*_full*'));report.update(new_csvs_created=False,new_synthetic_oof_or_full_model_fits=False)
  ledger=json.loads((ROOT/'research_w3_submission_ledger.json').read_text());assert ledger['completed_new_submissions']==ledger['reserved_slots']==0 and ledger['remaining_unreserved_slots']==2;report['submission_usage']={'completed':0,'reserved':0,'remaining':2,'parent_only':True}
 report.update(status='passed',completed_utc=now(),limitation='Reused partitions and adaptively chosen method family; conditional intervals exclude selection uncertainty. Positive point estimates do not satisfy the frozen confirmation gate.');dump(ROOT/'research_w3_closeout_audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()

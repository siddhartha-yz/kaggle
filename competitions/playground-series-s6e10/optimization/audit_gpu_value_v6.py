"""Independent read-only candidate/OOF/training-map audit; no training or submission."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="6"
import numpy as np,pandas as pd,joblib,cloudpickle,torch
torch.set_num_threads(6);torch.set_num_interop_threads(1)
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sel=json.loads((ROOT/'research_gpu_value_selection.json').read_text());assert sel['eligible_for_full'] and sel['chosen_variant']=='raw' and sel['n_ens']==2
 train=pd.read_csv(ROOT/'data/train.csv');raw=train.drop(columns=['id','satisfaction']);y=train.satisfaction.astype('int8').to_numpy();z=np.load(ROOT/'research_artifacts/gpu_value_raw_oof_n2/oof.npz');v=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert all(np.array_equal(z[k],v[k]) for k in ['id','target','fold']);assert np.array_equal(z['id'],train.id) and np.array_equal(z['target'],y);candidate=.75*v['pred']+.25*z['pred'];assert np.array_equal(candidate,z['candidate']);comp=compare(y,v['pred'],candidate);assert comp['delta_auc']>=.0002 and comp['conditional_95pct_ci'][0]>0
 foldaudits=[];assembled=np.full(len(y),np.nan)
 for k in range(5):
  folder=ROOT/'research_artifacts/gpu_value_raw_oof_n2'/f'fold{k}';r=json.loads((folder/'result.json').read_text());p=np.load(folder/'validation.npz');fi=np.flatnonzero(z['fold']!=k);vi=np.flatnonzero(z['fold']==k);assert np.array_equal(p['idx'],vi) and np.array_equal(p['id'],train.id.iloc[vi]) and np.array_equal(p['target'],y[vi]);assembled[vi]=p['pred'];tr=joblib.load(folder/'transformer.joblib');assert tr.variant=='raw' and not hasattr(tr,'context') and len(tr.output_columns)==38;medians=raw.iloc[fi][tr.numeric].median();pd.testing.assert_series_equal(tr.medians,medians);assert r['actual_epochs']==3 and r['model_reload_max_abs_diff']<1e-7 and all(t['num_threads']<=6 for t in r['threadpools'])
  # Model stores CPU tensors; verify every learned category vocabulary against this fold's training rows.
  assert sha(folder/'model.pkl')==r['model_sha256']
  with (folder/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
  assert all(p.device.type=='cpu' for p in model.cv_alg_interface_.model.parameters())
  converter=model.x_converter_;catcols=list(converter.cat_tf.transformers_[0][2]);enc=converter.cat_tf.named_transformers_['categorical'];assert len(catcols)==len(enc.categories_)==21
  for c,actual in zip(catcols,enc.categories_):
   if c.startswith('value_cat_'):
    source=tr.numeric[int(c.rsplit('_',1)[1])];expected=np.sort(raw.iloc[fi][source].fillna(tr.medians[source]).astype('int64').astype(str).unique())
   else:expected=np.sort(raw.iloc[fi][c].astype(str).unique())
   assert np.array_equal(actual,expected),c
  fc=compare(y[vi],v['pred'][vi],p['candidate']);assert fc['delta_auc']>0
  foldaudits.append({'fold':k,'train_rows':len(fi),'valid_rows':len(vi),'training_only_medians_exact':True,'training_only_category_vocabularies_exact':True,'raw_feature_width':38,'category_columns':21,'model_reload_max_abs_diff':r['model_reload_max_abs_diff'],'comparison':fc})
 assert np.array_equal(assembled,z['pred'])
 full=ROOT/'research_artifacts/gpu_value_v6_full';fr=json.loads((full/'result.json').read_text());assert fr['status']=='complete';parts=np.load(full/'test_components.npz');csv=ROOT/'submission_realmlp_value_v6.csv';back=pd.read_csv(csv);sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert list(back)==['id','satisfaction'] and len(back)==299844 and back.id.is_unique and back.id.equals(sample.id) and np.array_equal(parts['id'],back.id);assert np.isfinite(back.satisfaction).all() and back.satisfaction.between(0,1).all();pred=.75*parts['v5']+.25*parts['realmlp'];assert np.array_equal(pred,parts['candidate']);maxdiff=float(np.max(np.abs(back.satisfaction-pred)));assert maxdiff<1e-14;assert sha(csv)==fr['submission']['sha256'];assert fr['model_reload_max_abs_diff']<1e-7 and fr['peak_reserved_bytes']<=2*1024**3 and fr['actual_epochs']==3
 out={'status':'verified','verified_utc':datetime.now(timezone.utc).isoformat(),'filename':csv.name,'sha256':sha(csv),'bytes':csv.stat().st_size,'rows':len(back),'columns':list(back),'id_order_matches_official_sample':True,'finite_range_0_1':True,'csv_component_max_abs_diff':maxdiff,'oof_vs_v5':comp,'fold_audits':foldaudits,'full_model_reload_max_abs_diff':fr['model_reload_max_abs_diff'],'full_gpu_peak_reserved_bytes':fr['peak_reserved_bytes'],'gpu_allocator_cap_bytes':2*1024**3,'new_neural_view_uses_source_teacher_features':False,'v5_component_source_audit_preserved':True,'no_training_performed_by_this_audit':True,'limitation':'Partitions reused for selection; paired conditional intervals exclude selection uncertainty.'};(ROOT/'submission_realmlp_value_v6_independent_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='fold_audits'},indent=2))
if __name__=='__main__':main()

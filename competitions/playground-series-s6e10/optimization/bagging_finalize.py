"""Bind complete frozen models and independently audit the second CSV."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,subprocess,sys
os.environ.setdefault('OMP_NUM_THREADS','6');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/kaggle-s6e10-mpl')
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
plan=json.loads((root/'bagging_plan.json').read_text());result=json.loads((root/'bagging_result.json').read_text())
assert datetime.now(timezone.utc)<datetime.fromisoformat(plan['delivery_cutoff_utc'])
assert result['decision']=='eligible_for_full_refit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'best_ensemble.json').read_text());manifest['frozen_utc']=plan['frozen_utc'];manifest['created_utc']=datetime.now(timezone.utc).isoformat()
for component in manifest['components']:component['weight']=.4
folder=Path('iteration_artifacts/lgb63cats_bagging_full_s20261017');r=json.loads((root/folder/'result.json').read_text())
assert r['status']=='complete' and r['split']=='full' and r['train_rows']==699635 and r['valid_rows']==0
assert r['fixed_trees']==r['best_iterations']==639
manifest['components'].append({'directory':str(folder),'weight':.2,'trees':639,'model_sha256':sha(root/folder/'model.txt')})
with (root/'bagging_ensemble.json').open('x') as f:json.dump(manifest,f,indent=2)
out=root/'submission_cb8_lgb63cats_bagged_v2.csv';assert not out.exists()
subprocess.run([sys.executable,str(root/'predict_best.py'),'--manifest','bagging_ensemble.json','--output',out.name],cwd=root,check=True)
with (root/'data/test.csv').open(newline='') as f:ids=[row['id'] for row in csv.DictReader(f)]
with (root/'data/sample_submission.csv').open(newline='') as f:
 reader=csv.DictReader(f);cols=reader.fieldnames;sids=[row['id'] for row in reader]
with out.open(newline='') as f:
 reader=csv.DictReader(f);outcols=reader.fieldnames;rows=list(reader)
assert outcols==cols==['id','satisfaction'] and len(rows)==299844
assert [r['id'] for r in rows]==ids==sids and len(set(ids))==299844
pred=np.array([float(r['satisfaction']) for r in rows]);assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
expected=np.zeros(len(ids))
for path,w in [('iteration_artifacts/cb8_full_s42/submission_cb8.csv',.4),('iteration_artifacts/lgb63cats_full_s42/submission_lgb63cats.csv',.4),(str(folder/'submission_bagged_component.csv'),.2)]:
 frame=pd.read_csv(root/path);assert frame.id.astype(str).tolist()==ids;expected+=w*frame.satisfaction.to_numpy()
assert np.allclose(pred,expected,rtol=0,atol=1e-15)
assert sha(root/'submission_cb8_lgb63cats_frozen.csv')==plan['current_best_sha256']
report={'created_utc':datetime.now(timezone.utc).isoformat(),'filename':out.name,'sha256':sha(out),'bytes':out.stat().st_size,'rows':len(rows),'columns':outcols,'exact_lexical_id_order':True,'unique_ids':True,'finite_probability_in_range':True,'minimum':float(pred.min()),'maximum':float(pred.max()),'mean':float(pred.mean()),'readback_verified':True,'max_difference_from_saved_full_components':float(np.max(np.abs(pred-expected))),'full_models_only':True,'actual_trees':[1308,639,639],'weights':[.4,.4,.2],'manifest_sha256':sha(root/'bagging_ensemble.json'),'previous_best_unchanged':True,'formal_submissions_total_before_parent_action':1,'remaining_submission_maximum':1,'no_local_login_or_submission':True,'comparisons':result['comparisons']}
assert datetime.now(timezone.utc)<datetime.fromisoformat(plan['delivery_cutoff_utc'])
with (root/'bagging_submission_audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))

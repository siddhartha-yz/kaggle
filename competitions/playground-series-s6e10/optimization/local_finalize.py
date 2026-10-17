"""Bind full CPU models to the frozen recipe and independently audit the CSV."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
recipe=json.loads((root/'frozen_recipe.json').read_text())
preflight=json.loads((root/'local_preflight.json').read_text())
confirmation=json.loads((root/'local_confirmation.json').read_text())
assert confirmation['decision']=='proceed_full_frozen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'frozen_utc':recipe['frozen_utc'],'created_utc':datetime.now(timezone.utc).isoformat(),'data_sha256':preflight['data_sha256'],'positive_class':'True','components':[]}
for c in recipe['components']:
    directory=Path('iteration_artifacts')/f"{c['experiment']}_full_s42"
    r=json.loads((root/directory/'result.json').read_text())
    assert r['status']=='complete' and r['split']=='full'
    assert r['fixed_trees']==r['best_iterations']==c['trees']
    assert r['train_rows']==699635 and r['valid_rows']==0
    model=root/directory/('model.cbm' if r['config']['kind']=='catboost' else 'model.txt')
    manifest['components'].append({'directory':str(directory),'weight':c['weight'],'trees':c['trees'],'model_sha256':sha(model)})
with (root/'best_ensemble.json').open('x') as f:json.dump(manifest,f,indent=2)
output=root/'submission_cb8_lgb63cats_frozen.csv'
assert not output.exists()
subprocess.run([sys.executable,str(root/'predict_best.py'),'--output',output.name],cwd=root,check=True)
# Independent CSV audit against official lexical ID order and component predictions.
with (root/'data/test.csv').open(newline='') as f:
    reader=csv.DictReader(f);test_ids=[row['id'] for row in reader]
with (root/'data/sample_submission.csv').open(newline='') as f:
    reader=csv.DictReader(f);cols=reader.fieldnames;sample_ids=[row['id'] for row in reader]
with output.open(newline='') as f:
    reader=csv.DictReader(f);outcols=reader.fieldnames;rows=list(reader)
assert outcols==cols==['id','satisfaction']
ids=[r['id'] for r in rows];assert ids==test_ids==sample_ids
assert len(rows)==299844 and len(set(ids))==299844
prob=np.array([float(row['satisfaction']) for row in rows])
assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
expected=np.zeros(len(rows))
for c in recipe['components']:
    component=pd.read_csv(root/'iteration_artifacts'/f"{c['experiment']}_full_s42"/f"submission_{c['experiment']}.csv")
    assert component.id.astype(str).tolist()==ids
    expected+=c['weight']*component.satisfaction.to_numpy()
assert np.allclose(prob,expected,rtol=0,atol=1e-15)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'filename':output.name,'sha256':sha(output),'bytes':output.stat().st_size,'rows':len(rows),'columns':outcols,'id_order_exact_match':True,'id_unique':True,'finite_and_in_range':True,'minimum':float(prob.min()),'maximum':float(prob.max()),'mean':float(prob.mean()),'max_difference_from_saved_full_components':float(np.max(np.abs(prob-expected))),'readback_verified':True,'full_models_only':True,'manifest_sha256':sha(root/'best_ensemble.json'),'local_new_formal_submissions':0,'transfer_instruction':'Parent coordinates at most 2 total new formal submissions. No local Kaggle login, upload, or submission was performed.'}
with (root/'submission_audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))

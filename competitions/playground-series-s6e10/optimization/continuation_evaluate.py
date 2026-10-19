"""One pre-frozen complementary-model test; no public-score tuning."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
import numpy as np
from compare_validation import compare
root=Path(__file__).resolve().parent
plan=json.loads((root/'continuation_plan.json').read_text())
deadline=datetime.fromisoformat(plan['deadline_utc'])
assert datetime.now(timezone.utc)<deadline
assert hashlib.sha256((root/'submission_cb8_lgb63cats_frozen.csv').read_bytes()).hexdigest()==plan['current_best_csv_sha256']
report={'plan':plan,'started_utc':datetime.now(timezone.utc).isoformat(),'comparisons':{}}
def assess(split):
    arrays={name:np.load(root/'iteration_artifacts'/f'{name}_{split}_s42/validation.npz') for name in plan['candidate']}
    ref=arrays['cb8']
    for v in arrays.values():
        for key in ['idx','id','target']:assert np.array_equal(v[key],ref[key])
        assert np.isfinite(v['pred']).all() and ((v['pred']>=0)&(v['pred']<=1)).all()
    current=.5*arrays['cb8']['pred']+.5*arrays['lgb63cats']['pred']
    candidate=sum(plan['candidate'][name]['weight']*v['pred'] for name,v in arrays.items())
    r=compare(ref['target'],current,candidate)
    r['comparison']='Incremental change relative to submitted frozen 50/50 blend, not historical single-model baseline'
    return r
report['comparisons']['primary']=assess('primary')
(root/'continuation_progress.json').write_text(json.dumps(report,indent=2)+'\n')
print('PRIMARY',json.dumps(report['comparisons']['primary']),flush=True)
# One short fixed-tree confirmation fit, including negative results; no early stopping.
folder=root/'iteration_artifacts/lgb63allcats_confirm_s42'
assert not folder.exists()
assert datetime.now(timezone.utc)<deadline
remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
with (root/'continuation_lgb63allcats_confirm.log').open('x') as log:
    subprocess.run([sys.executable,'-u',str(root/'iterate.py'),'--experiment','lgb63allcats','--split','confirm','--trees','415'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
record=json.loads((folder/'result.json').read_text())
assert record['status']=='complete' and record['best_iterations']==record['fixed_trees']==415
report['comparisons']['confirm']=assess('confirm')
a=report['comparisons']['primary'];b=report['comparisons']['confirm']
report['decision']='eligible_for_full_refit' if a['delta_auc']>0 and b['delta_auc']>0 and b['conditional_95pct_ci'][0]>0 else 'retain_submitted_best'
report['ended_utc']=datetime.now(timezone.utc).isoformat()
report['caveat']='Development model selection and reused confirmation samples limit independent interpretation; conditional intervals exclude selection uncertainty.'
with (root/'continuation_result.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)

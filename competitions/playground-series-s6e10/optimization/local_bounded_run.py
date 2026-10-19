"""Run only explicitly frozen CPU recipes before the authorized deadline."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parent
deadline=datetime(2026,10,17,18,47,49,tzinfo=timezone.utc)
p=argparse.ArgumentParser();p.add_argument('stage',choices=['confirm','full']);a=p.parse_args()
assert datetime.now(timezone.utc)<deadline,'Training budget expired'
assert (root/'local_preflight.json').is_file(),'Preflight must pass first'
plan=root/'frozen_recipe.json'
if not plan.exists():
    assert a.stage=='confirm'
    recipe={'frozen_utc':datetime.now(timezone.utc).isoformat(),'deadline_utc':deadline.isoformat(),'positive_class':'True','model_seed':42,'confirm_split_seed':20261017,'confirm_baseline':{'experiment':'cbbase','trees':956},'components':[{'experiment':'cb8','trees':1308,'weight':.5},{'experiment':'lgb63cats','trees':639,'weight':.5}],'decision_rule':'Proceed to full only if frozen 50/50 confirm blend paired AUC difference has positive lower conditional 95% bound; no tuning on confirm. This reused-data stability split is not an independent test.'}
    with plan.open('x') as f:json.dump(recipe,f,indent=2)
recipe=json.loads(plan.read_text())
if a.stage=='full':
    report=json.loads((root/'local_confirmation.json').read_text())
    assert report['decision']=='proceed_full_frozen'
runs=([recipe['confirm_baseline']] if a.stage=='confirm' else [])+recipe['components']
for item in runs:
    assert datetime.now(timezone.utc)<deadline,'Training budget expired'
    folder=root/'iteration_artifacts'/f"{item['experiment']}_{a.stage}_s42"
    assert not folder.exists(),f'Refuse overwrite: {folder}'
    logfile=root/f"local_{item['experiment']}_{a.stage}.log"
    remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    with logfile.open('x') as log:
        command=[sys.executable,'-u',str(root/'iterate.py'),'--experiment',item['experiment'],'--split',a.stage,'--trees',str(item['trees'])]
        print(datetime.now(timezone.utc).isoformat(),'START',command,flush=True)
        try:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        except subprocess.TimeoutExpired:raise SystemExit('Deadline reached; stopped child training.')
    record=json.loads((folder/'result.json').read_text())
    assert record['status']=='complete' and record['best_iterations']==item['trees']
    print(datetime.now(timezone.utc).isoformat(),'COMPLETE',item['experiment'],record.get('validation_auc'),record['elapsed_seconds'],flush=True)

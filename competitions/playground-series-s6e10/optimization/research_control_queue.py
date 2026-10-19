from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
root=Path(__file__).resolve().parent;p=json.loads((root/'research_protocol.json').read_text())
plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'reason':'All high-cardinality distance-category variants underperformed; old-model predictions confirm base preprocessing equivalence. Isolate whether category splitting masks useful numeric group statistics or properly cross-fitted target signal.','variants':['distance_stats_numeric','distance_te_only'],'trees':639,'no_weight_search':True}
(root/'research_control_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for mode in plan['variants']:
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 remaining=(datetime.fromisoformat(p['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds()
 with (root/f'research_b1_{mode}.log').open('x') as log:
  subprocess.run([sys.executable,'-u',str(root/'research_run.py'),'--mode',mode,'--tag','b1'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 r=json.loads((root/'research_artifacts'/f'b1_{mode}_primary/result.json').read_text())
 print(mode,r['validation_auc'],r['vs_existing_lgb63cats']['delta_auc'],r['elapsed_seconds'],flush=True)

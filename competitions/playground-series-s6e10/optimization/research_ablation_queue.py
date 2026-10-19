from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
root=Path(__file__).resolve().parent;p=json.loads((root/'research_protocol.json').read_text())
for mode in ['distance_category','distance_frequency','distance_group_stats']:
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 remaining=(datetime.fromisoformat(p['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds()
 with (root/f'research_a1_{mode}.log').open('x') as log:
  subprocess.run([sys.executable,'-u',str(root/'research_run.py'),'--mode',mode],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 r=json.loads((root/'research_artifacts'/f'a1_{mode}_primary/result.json').read_text())
 print(mode,r['validation_auc'],r['vs_existing_lgb63cats']['delta_auc'],r['vs_submitted_ensemble']['delta_auc'],r['elapsed_seconds'],flush=True)

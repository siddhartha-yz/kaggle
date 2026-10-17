from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
r=Path(__file__).resolve().parent;p=json.loads((r/'research_protocol.json').read_text())
commands=[['--mode','distance_stats_numeric','--split','confirm','--tag','b1'],['--mode','distance_stats_te','--tag','c1'],['--mode','distance_stats_numeric','--tag','c2','--trees','4000','--early-stop']]
(r/'research_promotion_plan.json').write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),'reason':'Numeric distance groups materially beat the existing submitted model; first check fixed-tree confirmation, measure incremental inner-OOF TE, then test whether 639 rounds underfit the stronger numeric-group representation. No parameter grid.','commands':commands},indent=2)+'\n')
for args in commands:
 assert datetime.now(timezone.utc)<datetime.fromisoformat(p['training_start_cutoff_utc'])
 mode=args[1];tag=args[args.index('--tag')+1];split=args[args.index('--split')+1] if '--split' in args else 'primary'
 with (r/f'research_{tag}_{mode}_{split}.log').open('x') as log:
  subprocess.run([sys.executable,'-u',str(r/'research_run.py'),*args],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=(datetime.fromisoformat(p['training_hard_cutoff_utc'])-datetime.now(timezone.utc)).total_seconds())
 result=json.loads((r/'research_artifacts'/f'{tag}_{mode}_{split}/result.json').read_text());print(tag,mode,split,result['validation_auc'],result['best_iterations'],result['elapsed_seconds'],flush=True)

"""Managed serial R22 worker -> CPU gates -> replication only if qualified."""
from pathlib import Path
import subprocess,json,sys,datetime
ROOT=Path(__file__).resolve().parent
r=None
for stage in ('fivefold','threefold'):
    subprocess.run([sys.executable,'-u','research_r22_worker.py',stage],cwd=ROOT,check=True)
    subprocess.run([str(ROOT.parent/'.venv/bin/python'),'-u','research_r22_check.py',stage],cwd=ROOT,check=True)
    r=json.loads((ROOT/f'research_artifacts/r22_global_tabicl_{stage}/result.json').read_text())
    if not r['eligible']:break
v=dict(status='complete',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_stage=r['stage'],qualified=r['stage']=='threefold' and r['eligible'],csv_generated=False,formal_new_submissions=0,actual_top500_verified=False)
(ROOT/'research_r22_completion.json').write_text(json.dumps(v,indent=2)+'\n');print('R22_COMPLETION',v,flush=True)

"""Frozen full-pair configuration and task-only disk accounting; no quality scoring."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='6'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,threading
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts/r94_rtd_pair'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(path,value):
 path=Path(path);t=path.with_name('.'+path.name+f'.{os.getpid()}.{threading.get_native_id()}.tmp')
 t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(path)
def ownbytes():
 return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in (ROOT/'research_artifacts/r94_rtd_resource_probe').rglob('*') if p.is_file())+sum(p.stat().st_size for p in ROOT.glob('*r94*') if p.is_file())
def reserve(extra=8*2**20):
 assert shutil.disk_usage(ROOT).free>=2*2**30+extra,'Diskreserve2GiB'
 assert ownbytes()+extra<=192*2**20,'R94allartifacts192MiBcap'
def plan(inputs=False):
 p=json.loads((ROOT/'research_r94_full_plan.json').read_text())
 for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
 if inputs:
  for n,h in p['input_sha256'].items():assert sha(ROOT/n)==h,n
 return p

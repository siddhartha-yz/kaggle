from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts/r92_member_origins'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def artifacts_bytes():
 return sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())+sum(f.stat().st_size for f in ROOT.glob('research_r92*') if f.is_file())
def write_guard(estimate_MiB=1):
 assert shutil.disk_usage(ROOT).free>=2*2**30+estimate_MiB*2**20,'Diskreserve2GiBplusplannedwrite'
 assert artifacts_bytes()+estimate_MiB*2**20<=128*2**20,'TotalR92artifactcap128MiB'
def plan(all_inputs=False):
 p=json.loads((ROOT/'research_r92_plan.json').read_text())
 for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
 if all_inputs:
  for n,h in p['artifact_sha256'].items():assert sha(ROOT/n)==h,n
 return p
def refs(stage):
 tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');ce=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');rank=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz')
 for r in [tree,ce]:
  for k in ['id','target','fold']:assert np.array_equal(r[k],rank[k])
 assert np.max(np.abs(candidate(ce['candidate'],rank['pred'])-rank['candidate']))<1e-12
 return tree,ce,rank
def candidate(v11,q):return expit(.875*logit(np.clip(v11,1e-6,1-1e-6))+.125*logit(np.clip(q,1e-6,1-1e-6)))
def source(stage,k,p):
 src=ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/fold{k}';meta=p['models'][stage][k]
 for n,h in meta['files_sha256'].items():assert sha(src/n)==h,str(src/n)
 return src,meta

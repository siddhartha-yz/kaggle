from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts/r87_official_teacher_nn'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def verify():
 p=json.loads((ROOT/'research_r87_plan.json').read_text())
 for n,h in p['frozen_sha256'].items():assert sha(ROOT/n)==h,n
 return p
def refs(stage):
 z=[np.load(ROOT/f'research_artifacts/{n}_{stage}/oof.npz') for n in ['r21_probability_lgb_officialcb','r23_conditional_lean','r68_pure_ranknet']]
 for b in z[1:]:
  for k in ('id','target','fold'):assert np.array_equal(z[0][k],b[k]),k
 assert np.max(np.abs(candidate(z[0]['tree'],z[1]['pred'],z[2]['pred'])-z[2]['candidate']))<1e-12
 return z
def candidate(tree,ce,rank):
 clip=lambda x:np.clip(x,1e-6,1-1e-6)
 v11=expit(.5*logit(clip(tree))+.5*logit(clip(ce)))
 return expit(.875*logit(clip(v11))+.125*logit(clip(rank)))

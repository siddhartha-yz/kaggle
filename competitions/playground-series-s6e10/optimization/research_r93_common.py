"""Official train/test covariates only; no SAT labels or existing predictions loaded."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,threading,time,psutil,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts/r93_official_distribution_audit'
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
CATS=['Gender','Customer Type','Type of Travel','Class']
CONT=['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
FEATURES=['Age','Flight Distance',*RATINGS,'Departure Delay in Minutes','Arrival Delay in Minutes',*CATS]
NUM=[c for c in FEATURES if c not in CATS]
MISSING='__MISSING__'
HASH_KEY='kaggle-s6e10-av1'
assert len(FEATURES)==21 and len(HASH_KEY)==16 and not {'id','satisfaction','domain'}.intersection(FEATURES)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def ownbytes():return sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())+sum(f.stat().st_size for f in ROOT.glob('research_r93*') if f.is_file())
def reserve(MiB=1):
 assert shutil.disk_usage(ROOT).free>=2*2**30+MiB*2**20,'Diskreserve2GiBplusplannedwrite'
 assert ownbytes()+MiB*2**20<=32*2**20,'R93outputs32MiBcap'
def plan(all_inputs=False):
 p=json.loads((ROOT/'research_r93_plan.json').read_text())
 for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
 if all_inputs:
  for n,h in p['input_sha256'].items():assert sha(ROOT/n)==h,n
 return p
def load():
 blocks=[];ids=[]
 for name,n in [('train',699635),('test',299844)]:
  df=pd.read_csv(ROOT/f'data/{name}.csv',usecols=['id',*FEATURES],dtype={**{c:'float64' for c in NUM},**{c:'string' for c in CATS}})
  assert len(df)==n and set(df)==set(['id',*FEATURES]) and 'satisfaction' not in df
  ids.append(df['id'].to_numpy());x=df[FEATURES].copy()
  for c in CATS:x[c]=x[c].fillna(MISSING).astype(str)
  for c in NUM:
   assert not np.isinf(x[c].to_numpy()).any();x.loc[x[c]==0,c]=0.0
  blocks.append(x)
 return blocks[0],blocks[1],ids[0],ids[1]
def fingerprints(raw):return pd.util.hash_pandas_object(raw,index=False,hash_key=HASH_KEY).to_numpy('uint64')
def split(raw):
 h=fingerprints(raw);g=h^np.uint64(20261022)
 with np.errstate(over='ignore'):
  g=(g^(g>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);g=(g^(g>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);g=g^(g>>np.uint64(31))
 hold=g%np.uint64(10)<3;fi=np.flatnonzero(~hold);vi=np.flatnonzero(hold);np.random.default_rng(20261025).shuffle(fi)
 assert not np.intersect1d(h[fi],h[vi]).size
 return fi,vi,h

def monitor(stop,path):
 from research_resume_policy_20261021 import check_training
 while not stop.wait(.5):
  try:
   check_training();reserve();rss=psutil.Process().memory_info().rss;available=psutil.virtual_memory().available
   assert rss<=6*2**30 and available>=4*2**30
  except BaseException as e:dump(path,dict(reason=repr(e),utc=now()));os._exit(86)
def guard():
 reserve();assert psutil.virtual_memory().available>=6*2**30;OUT.mkdir(parents=True,exist_ok=True);stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,OUT/'resource_failure.json'),daemon=True);thread.start();return stop,thread

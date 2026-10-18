"""Pin official wheels in a separate venv, inherit fixed CUDA runtime, fetch public hashed weights."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,sys,json,hashlib,subprocess,urllib.request
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version,InvalidVersion
ROOT=Path(__file__).absolute().parent;SRC=ROOT/'research_sources/tabicl';ENV=ROOT.parent/'.venv-tabicl-w3';GPU=ROOT.parent/'.venv-gpu';STATE=ROOT/'research_w3_tabicl_setup_state.json'
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def getjson(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Kaggle-local-research/1.0'}),timeout=30) as r:return json.load(r)
def main():
 SRC.mkdir(exist_ok=True);rec={'status':'creating_isolated_environment','started_utc':now(),'new_environment':str(ENV),'inherited_readonly_CUDA_packages':str(GPU/'lib/python3.12/site-packages'),'global_or_GPU_environment_modified':False};dump(STATE,rec)
 try:
  assert not ENV.exists(),'Refuseoverwritingexistingvenv';subprocess.run([os.path.abspath(GPU/'bin/python'),'-m','venv',str(ENV)],check=True);site=ENV/'lib/python3.12/site-packages';(site/'gpu_parent_packages.pth').write_text(str(GPU/'lib/python3.12/site-packages')+'\n');py=os.path.abspath(ENV/'bin/python');prefix=subprocess.check_output([py,'-c','import sys;print(sys.prefix)'],text=True).strip();assert prefix==str(ENV);tagrank={t:i for i,t in enumerate(sys_tags())};cutoff=datetime.now(timezone.utc)-timedelta(days=7);wheels=SRC/'wheels';wheels.mkdir(exist_ok=True);records=[]
  packages={'tabicl':'2.2.0','einops':'0.8.1','huggingface-hub':'0.36.0','requests':'2.32.5','urllib3':None,'certifi':None,'charset-normalizer':None,'idna':None,'hf-xet':'1.1.10'}
  for name,pinned in packages.items():
   meta=getjson(f'https://pypi.org/pypi/{name}/json');versions=[pinned] if pinned else sorted([v for v in meta['releases'] if not Version(v).is_prerelease],key=Version,reverse=True);chosen=None
   for version in versions:
    m=getjson(f'https://pypi.org/pypi/{name}/{version}/json');possible=[]
    for u in m['urls']:
     if u.get('yanked') or not u['filename'].endswith('.whl'):continue
     if datetime.fromisoformat(u['upload_time_iso_8601'].replace('Z','+00:00'))>cutoff:continue
     tags=parse_wheel_filename(u['filename'])[3];ranks=[tagrank[t] for t in tags if t in tagrank]
     if ranks:possible.append((min(ranks),u))
    if possible:chosen=min(possible,key=lambda x:x[0])[1];break
   assert chosen is not None,name;assert chosen['url'].startswith('https://files.pythonhosted.org/');dest=wheels/chosen['filename']
   with urllib.request.urlopen(chosen['url'],timeout=30) as r:b=r.read(100*1024**2+1)
   assert len(b)<=100*1024**2 and hashlib.sha256(b).hexdigest()==chosen['digests']['sha256'];dest.write_bytes(b);records.append({'package':name,'version':version,'filename':dest.name,'path':str(dest),'url':chosen['url'],'sha256':sha(dest),'bytes':len(b)});rec.update(status='fetching_pinned_wheels',wheels=records);dump(STATE,rec);print('WHEEL',name,version,len(b),flush=True)
  rec['status']='installing_local_wheels';dump(STATE,rec);subprocess.run([py,'-m','pip','--isolated','install','--no-deps','--no-index','--no-cache-dir',*[r['path'] for r in records]],check=True);subprocess.run([py,'-m','pip','check'],check=True)
  hf=json.loads((SRC/'hf_official_metadata.json').read_text());assert hf['gated'] is False and not hf['private'] and hf['license']=='bsd-3-clause';name='tabicl-classifier-v2-20260212.ckpt';info=next(x for x in hf['siblings'] if x['rfilename']==name);dest=SRC/name;url=f"https://huggingface.co/jingang/TabICL/resolve/{hf['sha']}/{name}?download=true";rec.update(status='fetching_public_weights',model_expected_bytes=info['size']);dump(STATE,rec);h=hashlib.sha256();size=0
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Kaggle-local-research/1.0'}),timeout=60) as response,(SRC/(name+'.part')).open('wb') as f:
   while True:
    b=response.read(1024**2)
    if not b:break
    size+=len(b);assert size<=info['size'];h.update(b);f.write(b)
  assert size==info['size'] and h.hexdigest()==info['lfs']['sha256'];(SRC/(name+'.part')).rename(dest);rec.update(status='complete',completed_utc=now(),model={'filename':name,'path':str(dest),'repo':'jingang/TabICL','revision':hf['sha'],'sha256':h.hexdigest(),'bytes':size,'license':hf['license'],'gated':False,'public_url':url},wheels=records);dump(STATE,rec);dump(ROOT/'research_w3_tabicl_dependencies.json',rec);print('SETUP_COMPLETE',now(),flush=True)
 except BaseException as e:rec.update(status='failed',error=repr(e),ended_utc=now());dump(STATE,rec);raise
if __name__=='__main__':main()

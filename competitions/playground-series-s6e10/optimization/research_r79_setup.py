"""Dependency-only TabSTAR preparation; never instantiate/download a model."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parent
def state(status,**kw):
 d=dict(status=status,checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),classifier_fits=0,model_weights_downloaded=False,**kw);(ROOT/'research_r79_setup_state.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def main():
 wheel=ROOT/'research_sources/new_foundation_feasibility/tabstar-1.2.2-py3-none-any.whl';assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='c632d3ad0e389303cc787b9fcaf9ca50cfab4eb0143f03f2c631d1c678b455ba'
 env=dict(os.environ,UV_CACHE_DIR=str(ROOT/'research_artifacts/r79_uv_cache'),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HOME=str(ROOT/'research_artifacts/r79_hf_offline'),TOKENIZERS_PARALLELISM='false')
 python=ROOT.parent/'.venv-tabstar-r79/bin/python'
 state('dependency_install_running',venv='.venv-tabstar-r79')
 # Explicit target only: never uninstall/modify read-only borrowed environments.
 target=ROOT.parent/'.venv-tabstar-r79/lib/python3.12/site-packages'
 subprocess.run(['/home/yang-zhi/.local/bin/uv','pip','install','--python',str(python),'--target',str(target),'--no-deps','--index-url','https://pypi.org/simple',str(wheel),'transformers==4.49.0','peft==0.15.1','accelerate==1.5.2','skrub==0.5.1','tokenizers==0.21.0','regex==2024.11.6','scikit-learn==1.5.2'],env=env,check=True)
 code="""
import importlib.metadata as md,json
from packaging.requirements import Requirement
names=['tabstar','transformers','peft','accelerate','skrub','tokenizers','regex','scikit-learn']
bad=[]
for name in names:
 for raw in md.requires(name) or []:
  q=Requirement(raw)
  if q.marker and not q.marker.evaluate({'extra':''}):continue
  try:v=md.version(q.name)
  except md.PackageNotFoundError:bad.append(str(q));continue
  if q.specifier and v not in q.specifier:bad.append(str(q)+' actual='+v)
assert not bad,bad
from tabstar.tabstar_model import TabSTARClassifier
assert md.version('torch')=='2.9.1+cu128'
assert md.version('scikit-learn')=='1.5.2'
print('DEPENDENCY_AND_CLASS_IMPORT_PASSED',json.dumps({n:md.version(n) for n in names+['torch','numpy','pandas','scikit-learn','huggingface-hub','safetensors']}),flush=True)
"""
 subprocess.run([str(python),'-B','-c',code],env=env,check=True)
 state('passed_dependency_and_import_only',original_Torch_CUDA_unchanged=True,models_instantiated=0,GPU_used=False)
if __name__=='__main__':
 try:main()
 except BaseException as error:state('failed',error=repr(error));raise

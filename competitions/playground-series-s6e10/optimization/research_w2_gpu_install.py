"""Isolated official CUDA runtime installation with disk-backed recovery state."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,json,os
ROOT=Path(__file__).resolve().parent;gpu=Path(os.path.abspath(ROOT/'../.venv-gpu/bin/python'))
actual_prefix=subprocess.check_output([str(gpu),'-c','import sys; print(sys.prefix)'],text=True).strip()
assert Path(actual_prefix)==gpu.parent.parent, 'Refuse installation outside the GPU venv'
def state(stage,status,**kw):(ROOT/'research_w2_gpu_install_state.json').write_text(json.dumps(dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n')
steps=[('torch',[sys.executable,'-m','pip','--python',str(gpu),'install','--index-url','https://download.pytorch.org/whl/cu128','--report',str(ROOT/'research_sources/realmlp_official/torch_gpu_install.json'),'torch==2.9.1+cu128']),('base',[sys.executable,'-m','pip','--python',str(gpu),'install','--no-deps','--index-url','https://pypi.org/simple','--report',str(ROOT/'research_sources/realmlp_official/gpu_base_install.json'),'-r',str(ROOT/'requirements-realmlp-gpu-base.txt')])]
try:
 with (ROOT/'research_w2_gpu_install.log').open('w') as f:
  for name,args in steps:state(name,'running');print('START',name,flush=True);subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True);print('DONE',name,flush=True)
 state('complete','complete')
except BaseException as e:state('failed','failed',error=repr(e));raise

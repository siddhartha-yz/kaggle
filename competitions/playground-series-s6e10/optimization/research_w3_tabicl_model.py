"""Offline official TabICL with immutable-weight reuse; context preprocessing/cache still refit."""
from pathlib import Path
import os
TASK=Path(__file__).absolute().parent.parent
os.environ['HF_HOME']=str(TASK/'.cache-tabicl-w3/hf');os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';os.environ['TORCH_HOME']=str(TASK/'.cache-tabicl-w3/torch');os.environ['CUDA_CACHE_PATH']=str(TASK/'.cache-tabicl-w3/cuda');os.environ['TRITON_CACHE_DIR']=str(TASK/'.cache-tabicl-w3/triton')
from tabicl import TabICLClassifier
class FrozenTabICL(TabICLClassifier):
 def _load_model(self):
  if getattr(self,'_immutable_weights_loaded',False):return
  super()._load_model();self._immutable_weights_loaded=True
def config(root,n_estimators=4,cache=True):
 return dict(n_estimators=n_estimators,norm_methods=['none','power'],batch_size=1,kv_cache=cache,model_path=str(root/'research_sources/tabicl/tabicl-classifier-v2-20260212.ckpt'),allow_auto_download=False,checkpoint_version='tabicl-classifier-v2-20260212.ckpt',device='cuda:0',use_amp=True,use_fa3=False,offload_mode=False,random_state=42,n_jobs=6,verbose=False)

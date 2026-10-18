"""Process-local, evaluation-only TabICL memory adapter; installed files untouched.
Native TabICL2.2 BSD-3 implementation: soda-inria/tabicl. Keep KV projections
on CPU, stage one attention layer, chunk pointwise FFNs, and report the actual
process allocator allowance to native auto-batching. Attention math unchanged.
"""
from contextlib import contextmanager
import torch
from tabicl._model.inference import InferenceManager
from tabicl._model.kv_cache import KVCache
from tabicl._model import layers
CAP=778043392
CEILING=1073741824
class CPUEntries(dict):
    def __init__(self,source=None):
        super().__init__()
        if source is not None:self.update(source)
    def __setitem__(self,key,entry):super().__setitem__(key,entry.to('cpu'))
    def update(self,source=(),**kwargs):
        for k,v in dict(source,**kwargs).items():self[k]=v

def inference_config():
    common=dict(offload='cpu',safety_factor=.5,max_pinned_memory_mb=64,use_async=False,cpu_safety_factor=.5)
    return {k:dict(common) for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']}

@contextmanager
def bounded_offload(ff_chunk_rows=4096):
    old_memory=InferenceManager.get_available_gpu_memory;old_init=KVCache.__init__;old_to=KVCache.to
    old_attention=layers.multi_head_attention_forward;old_ff=layers.MultiheadAttentionBlock._ff_block
    counters=dict(staged_cache_entries=0,chunked_ff_calls=0)
    def memory(self):
        free=old_memory(self)
        if self.exe_device.type=='cuda':
            remaining=max(1.,(CAP-torch.cuda.memory_allocated()-32*2**20)/2**20)
            return min(free,remaining)
        return free
    def initialize(self,kv=None):old_init(self,kv=CPUEntries(kv))
    def to_cpu(self,device,dtype=None):return self.__class__(kv={k:v.to('cpu',dtype=dtype) for k,v in self.kv.items()})
    def attention(*args,**kwargs):
        q=kwargs.get('query',args[0] if args else None);entry=kwargs.get('cached_kv')
        if entry is not None and entry.is_valid() and entry.key.device!=q.device:
            kwargs['cached_kv']=entry.to(q.device);counters['staged_cache_entries']+=1
        return old_attention(*args,**kwargs)
    def ff(self,x):
        assert not self.training,'Memory adapter is evaluation-only'
        if x.shape[-2]<=ff_chunk_rows:return old_ff(self,x)
        counters['chunked_ff_calls']+=1
        return torch.cat([old_ff(self,piece) for piece in x.split(ff_chunk_rows,dim=-2)],dim=-2)
    InferenceManager.get_available_gpu_memory=memory;KVCache.__init__=initialize;KVCache.to=to_cpu
    layers.multi_head_attention_forward=attention;layers.MultiheadAttentionBlock._ff_block=ff
    try:yield counters
    finally:
        InferenceManager.get_available_gpu_memory=old_memory;KVCache.__init__=old_init;KVCache.to=old_to
        layers.multi_head_attention_forward=old_attention;layers.MultiheadAttentionBlock._ff_block=old_ff

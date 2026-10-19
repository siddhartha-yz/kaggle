"""Process-local v2 inference adapter: independent column attention, CPU KV storage.

Built with PriorLabs-TabPFN. Installed code/weights are unchanged. Cache build in
9.2 bypasses its ordinary attention batching, so cache mode otherwise exceeds
the allocator limit. Each column batch is independent; keep the same attention
and linear functions, moving completed KV entries to CPU before the next layer.
"""
from contextlib import contextmanager
import torch
from tabpfn.architectures import tabpfn_v2 as v2
from tabpfn.architectures.kv_cache import KVCacheEntry

@contextmanager
def streamed_cache():
    original_attention=v2.AlongColumnAttention.forward;original_to=v2.TabPFNV2Cache.to
    counters=dict(build_column_chunks=0,query_column_chunks=0,CPU_cache_moves=0)
    def attention(self,x_BcRE,single_eval_pos=None,*,cached_kv=None,return_kv=False):
        if not return_kv and cached_kv is None:
            return original_attention(self,x_BcRE,single_eval_pos)
        assert not self.training
        output=None;keys=[];values=[];grid_dtype=None
        for i in range(len(x_BcRE)):
            entry=None
            if cached_kv is not None:
                entry=KVCacheEntry(key=cached_kv.key[i:i+1].to(x_BcRE.device),value=cached_kv.value[i:i+1].to(x_BcRE.device),grid_dtype=cached_kv.grid_dtype);counters['query_column_chunks']+=1
            piece,generated=original_attention(self,x_BcRE[i:i+1],single_eval_pos,cached_kv=entry,return_kv=return_kv)
            if output is None:output=torch.empty((len(x_BcRE),*piece.shape[1:]),device=piece.device,dtype=piece.dtype)
            output[i:i+1].copy_(piece)
            if return_kv:
                assert generated is not None and generated.is_valid();cpu=generated.to('cpu');keys.append(cpu.key);values.append(cpu.value);grid_dtype=cpu.grid_dtype;counters['build_column_chunks']+=1;counters['CPU_cache_moves']+=1
            del piece,generated,entry
        result=KVCacheEntry(key=torch.cat(keys,dim=0),value=torch.cat(values,dim=0),grid_dtype=grid_dtype) if return_kv else None
        return output,result
    def cache_to(self,device):
        if torch.device(device).type!='cuda':return original_to(self,device)
        assert all(entry.key.device.type=='cpu' and entry.value.device.type=='cpu' for entry in self.kv.values())
        return v2.TabPFNV2Cache(kv=self.kv,feature_cache=self._dict_of_tensors_to(self.feature_cache,device),test_y_embedding=None if self.test_y_embedding is None else self.test_y_embedding.to(device),train_shape=self.train_shape)
    v2.AlongColumnAttention.forward=attention;v2.TabPFNV2Cache.to=cache_to
    try:yield counters
    finally:v2.AlongColumnAttention.forward=original_attention;v2.TabPFNV2Cache.to=original_to

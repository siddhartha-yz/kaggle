"""Process-local eval-only FFN output allocation; exact same fixed4096-row GEMMs."""
from contextlib import contextmanager
import torch
from tabicl._model import layers
from research_r22_tabicl_offload import bounded_offload
@contextmanager
def streamed_ffn():
    # Capture unpatched official pointwise FFN. Preserve R22 chunk sizes and math,
    # replacing retained output-list + cat by one preallocated output and copies.
    official=layers.MultiheadAttentionBlock._ff_block
    with bounded_offload(ff_chunk_rows=4096) as counters:
        prior=layers.MultiheadAttentionBlock._ff_block
        counters['preallocated_ff_calls']=0
        def ff(self,x):
            assert not self.training
            if x.shape[-2]<=4096:return official(self,x)
            counters['preallocated_ff_calls']+=1
            output=None
            for start in range(0,x.shape[-2],4096):
                piece=official(self,x[...,start:start+4096,:])
                if output is None:output=torch.empty(x.shape,device=piece.device,dtype=piece.dtype)
                output[...,start:start+piece.shape[-2],:].copy_(piece)
                del piece
            return output
        layers.MultiheadAttentionBlock._ff_block=ff
        try:yield counters
        finally:layers.MultiheadAttentionBlock._ff_block=prior

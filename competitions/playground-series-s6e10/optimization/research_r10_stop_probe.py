"""Verify a chosen stop preserves the nine-epoch schedule and stops at two."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,numpy as np,pandas as pd,torch
from research_w3_common import ROOT,now,dump
from research_r8_probe import BoundedRealMLP,CAP
from research_r10_streaming import validation_streaming_backend,ValidationStreamingModule
torch.set_num_threads(6);torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
rng=np.random.default_rng(42);X=pd.DataFrame(rng.normal(size=(4096,12)),columns=[f'x{j}' for j in range(12)]);y=rng.integers(0,2,len(X));cfg=dict(json.loads((ROOT/'research_r8_plan.json').read_text())['config']);cfg.update(n_epochs=9,stop_epoch=2,n_ens=2)
seen=[];original=ValidationStreamingModule.on_train_epoch_end
def hook(module):
 original(module);seen.append(int(module.current_epoch)+1);assert module.progress.max_epochs==9;assert all(p.device.type=='cuda' for p in module.parameters())
ValidationStreamingModule.on_train_epoch_end=hook
try:
 with validation_streaming_backend():model=BoundedRealMLP(**cfg).fit(X,y)
finally:ValidationStreamingModule.on_train_epoch_end=original
assert seen==[1,2] and not model.cv_alg_interface_.model.creator.is_cv
dump(ROOT/'research_r10_stop_probe.json',dict(status='passed',completed_utc=now(),observed_epochs=seen,schedule_max_epochs=9,official_data_used=False,actual_cuda_parameters_during_fit=True))

"""Frozen upstream TabSTAR raw21 LoRA adapter and observation-only tracing."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='6'
for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN']:os.environ.pop(k,None)
os.environ.update(HF_HOME=str(ROOT/'research_artifacts/r79_hf_offline'),HF_HUB_CACHE=str(ROOT/'research_artifacts/r79_hf_offline/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',MPLCONFIGDIR=str(ROOT/'research_artifacts/r79_matplotlib'),XDG_CACHE_HOME=str(ROOT/'research_artifacts/r79_runtime_cache'))
import hashlib,json,time
from contextlib import contextmanager
import numpy as np,torch
from tabstar.tabstar_model import TabSTARClassifier
from tabstar.training.trainer import TabStarTrainer

def parameters():
 return dict(is_paper_version=True,lora_lr=.001,lora_wd=0.,lora_r=32,lora_alpha=2,lora_dropout=.1,lora_batch=64,global_batch=128,max_epochs=3,patience=5,verbose=False,random_state=42,time_limit=7200,pretrain_dataset_or_path='alana89/TabSTAR',keep_model=True,val_batch_size=64)
def make_model(output_dir):return TabSTARClassifier(device=torch.device('cuda'),output_dir=str(output_dir),**parameters())
def configure():
 torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def digest(model,base_only=False):
 h=hashlib.sha256()
 for n,p in sorted(model.named_parameters()):
  if base_only and ('lora_A' in n or 'lora_B' in n):continue
  a=p.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
def predict(model,x):
 assert list(model.classes_)==['False','True'] and model.device.type=='cuda' and model.use_amp
 p=model.predict_proba(x);assert p.shape==(len(x),2) and np.isfinite(p).all() and np.max(np.abs(p.sum(axis=1)-1))<1e-6
 return p[:,1].astype('float64')
def preprocessor_state(p):
 def array(a):
  v=np.asarray(a);return dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist())
 return dict(y_name=p.y_name,y_values=p.y_values,constant_columns=p.constant_columns,date_columns=sorted(p.date_transformers),numerical={k:{a:array(getattr(v,a)) for a in ['mean_','var_','scale_','n_samples_seen_']} for k,v in sorted(p.numerical_transformers.items())},quantiles={k:{a:array(getattr(v,a)) for a in ['quantiles_','references_','n_quantiles_','n_features_in_']} for k,v in sorted(p.semantic_transformers.items())},target_classes=p.target_transformer.classes_.tolist(),d_output=p.d_output)

@contextmanager
def trace_native(callback,expected_base_hash):
 """Call original native methods unchanged; observe no RNG/gradient modifications."""
 original_epoch=TabStarTrainer._train_epoch;original_batch=TabStarTrainer._train_batch;original_eval=TabStarTrainer._evaluate_epoch;original_update=TabStarTrainer._do_update
 trace=dict(epochs=[],inner_evaluations=[],microbatches=0,updates_called=0,updates_executed=0)
 def batch(trainer,data):
  value=original_batch(trainer,data);trace['microbatches']+=1
  if trace['microbatches']%256==0:callback('native_GPU_training',microbatches=trace['microbatches'],epochs_completed=len(trace['epochs']),last_training_loss=float(value))
  return value
 def update(trainer):trace['updates_called']+=1;return original_update(trainer)
 def epoch(trainer,loader):
  if not trace['epochs']:
   trace['initial_base_hash']=digest(trainer.model,True);assert trace['initial_base_hash']==expected_base_hash
   trace['trainable_parameters']=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad);assert trace['trainable_parameters']==1597440
   trace['best_dir']=str(trainer.cp_manager.best_dir)
   def executed(optimizer,args,kwargs):trace['updates_executed']+=1
   trainer.optimizer.register_step_post_hook(executed)
  torch.cuda.synchronize();begin=time.monotonic();groups=[dict(lr=g['lr'],weight_decay=g['weight_decay'],wd_argument=g.get('wd')) for g in trainer.optimizer.param_groups];assert all(g['weight_decay']==.01 for g in groups)
  value=original_epoch(trainer,loader);torch.cuda.synchronize();assert np.isfinite(value)
  trace['epochs'].append(dict(rows=len(loader.dataset),microbatches=len(loader),seconds=time.monotonic()-begin,optimizer_groups=groups,training_loss=float(value),updates_called=trace['updates_called'],updates_executed=trace['updates_executed'],scaler_scale=float(trainer.scaler.get_scale())))
  callback('native_epoch_complete',epoch=len(trace['epochs']),seconds=trace['epochs'][-1]['seconds'],microbatches=trace['microbatches']);return value
 def evaluate(trainer,loader):
  value=original_eval(trainer,loader);assert all(np.isfinite(v) for v in value)
  trace['inner_evaluations'].append(dict(rows=len(loader.dataset),loss=float(value[0]),metric=float(value[1])))
  return value
 TabStarTrainer._train_epoch=epoch;TabStarTrainer._train_batch=batch;TabStarTrainer._evaluate_epoch=evaluate;TabStarTrainer._do_update=update
 try:yield trace
 finally:
  TabStarTrainer._train_epoch=original_epoch;TabStarTrainer._train_batch=original_batch;TabStarTrainer._evaluate_epoch=original_eval;TabStarTrainer._do_update=original_update

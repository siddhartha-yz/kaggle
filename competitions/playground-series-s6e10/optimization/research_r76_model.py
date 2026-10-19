"""Offline, unmodified official Mitra core; explicit train-only raw21 encoding."""
from pathlib import Path
import os,sys,hashlib,json
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_sources'))
os.environ['HF_HOME']=str(ROOT/'research_artifacts/r76_hf_offline')
import numpy as np,pandas as pd,torch
from safetensors.torch import load_file,save_file
from r76_mitra_core._internal.models.tab2d import Tab2D

class RawEncoder:
 def fit(self,raw):
  assert raw.shape[1]==21 and 'id' not in raw and 'satisfaction' not in raw
  self.columns=list(raw);self.cats=list(raw.select_dtypes(exclude='number'));self.nums=list(raw.select_dtypes(include='number'))
  assert len(self.cats)==4 and len(self.nums)==17
  self.vocabulary={c:sorted(raw[c].fillna('__MISSING__').astype(str).unique().tolist()) for c in self.cats}
  self.medians={c:float(raw[c].median()) for c in self.nums};assert all(np.isfinite(v) for v in self.medians.values())
  self.fit_rows=len(raw);return self
 def transform(self,raw):
  assert list(raw)==self.columns
  out=np.empty((len(raw),21),dtype='float32')
  for j,c in enumerate(self.columns):
   if c in self.vocabulary:out[:,j]=raw[c].fillna('__MISSING__').astype(str).map({v:k for k,v in enumerate(self.vocabulary[c])}).fillna(-1).to_numpy('float32')
   else:out[:,j]=raw[c].fillna(self.medians[c]).to_numpy('float32')
  assert np.isfinite(out).all();return out
 def state(self):return dict(columns=self.columns,categorical=self.cats,numerical=self.nums,vocabulary=self.vocabulary,numeric_medians=self.medians,fit_rows=self.fit_rows,unknown_category=-1)

def make_model(device='cuda',checkpoint=None):
 config=json.loads((ROOT/'research_sources/mitra_v2/config.json').read_text())
 model=Tab2D(**config,use_pretrained_weights=False,path_to_weights='',device=device)
 model.load_state_dict(load_file(str(checkpoint or ROOT/'research_sources/mitra_v2/model.safetensors'),device='cpu'),strict=True)
 assert not model.use_flash_attn
 return model.to(device)

def logits(model,X,y,V,device='cuda',bf16=True):
 # Copies because the official quantile embedding writes padding into support.
 S=torch.tensor(X,device=device,dtype=torch.float32).unsqueeze(0)
 Y=torch.tensor(y,device=device,dtype=torch.long).unsqueeze(0)
 Q=torch.tensor(V,device=device,dtype=torch.float32).unsqueeze(0)
 masks=[torch.zeros((1,n),device=device,dtype=torch.bool) for n in [X.shape[1],len(X),len(V)]]
 with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bf16 and device=='cuda'):
  out=model(S,Y,Q,*masks)[0].float()
 assert out.shape==(len(V),10) and torch.isfinite(out).all()
 return out

@torch.inference_mode()
def predict(model,X,y,V,device='cuda',bf16=True):
 model.eval();return logits(model,X,y,V,device,bf16)[:,:2].softmax(-1)[:,1].cpu().numpy().astype('float64')

def parameter_digest(model):
 d=hashlib.sha256()
 for name,t in sorted(model.state_dict().items()):d.update(name.encode());d.update(t.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()

def balanced_context(y,size,rng):
 classes=[np.flatnonzero(y==c) for c in [0,1]];assert all(len(v)>=size//2 for v in classes)
 selected=np.concatenate([rng.choice(v,size//2,replace=False) for v in classes]);rng.shuffle(selected);assert len(selected)==size and len(np.unique(selected))==size
 return selected

class SupportCache:
 """Exact fixed-context inference: retain each layer's original BF16 support K/V.

 The official support stream never attends to queries. Capture its original
 k/v projections without changing the core, then execute only the query stream.
 Cache stages on CPU so original prefill activations do not overlap GPU caches.
 """
 @torch.inference_mode()
 def __init__(self,model,X,y):
  self.model=model.eval();self.X=X.copy();self.y=y.copy();self.device=next(model.parameters()).device
  assert self.device.type=='cuda' and not model.use_flash_attn
  captured={};hooks=[]
  def capture(key):
   def hook(module,args,output):
    if key not in captured:captured[key]=output.cpu()
   return hook
  try:
   for i,layer in enumerate(model.layers):
    hooks.append(layer.attention1.k.register_forward_hook(capture((i,'k'))));hooks.append(layer.attention1.v.register_forward_hook(capture((i,'v'))))
   predict(model,X,y,X[:1])
  finally:
   for hook in hooks:hook.remove()
  assert len(captured)==2*len(model.layers)
  torch.cuda.empty_cache()
  self.kv=[]
  for i,layer in enumerate(model.layers):
   k=captured.pop((i,'k')).to(self.device);v=captured.pop((i,'v')).to(self.device);B,T,D=k.shape;H=layer.attention1.n_heads
   assert B==X.shape[1]+1 and T==len(X) and D==model.dim and k.dtype==torch.bfloat16 and v.dtype==torch.bfloat16
   self.kv.append((k.reshape(B,T,H,D//H).transpose(1,2),v.reshape(B,T,H,D//H).transpose(1,2)))
  self.bytes=sum((k.numel()*k.element_size()+v.numel()*v.element_size()) for k,v in self.kv)
 @torch.inference_mode()
 def predict(self,V):
  m=self.model;device=self.device;S=torch.tensor(self.X,device=device,dtype=torch.float32).unsqueeze(0);Q=torch.tensor(V,device=device,dtype=torch.float32).unsqueeze(0)
  ps=torch.zeros((1,len(self.X)),device=device,dtype=torch.bool);pf=torch.zeros((1,self.X.shape[1]),device=device,dtype=torch.bool)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   _,Q=m.x_quantile(S,Q,ps,pf);Q=m.x_embedding(Q)
   # Official Tab2DEmbeddingYClasses builds y_mask with zero integer indices.
   _,Y=m.y_embedding(torch.tensor(self.y,device=device,dtype=torch.long).unsqueeze(0),ps,len(V));query=torch.cat((Y,Q),dim=2)
   F=query.shape[2];D=query.shape[3]
   for layer,(k,v) in zip(m.layers,self.kv):
    residual=query;a=layer.layer_norm1(query).permute(0,2,1,3).reshape(F,len(V),D);H=layer.attention1.n_heads;q=layer.attention1.q(a).reshape(F,len(V),H,D//H).transpose(1,2)
    a=torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(F,len(V),D);a=layer.attention1.o(a);query=residual+a.reshape(1,F,len(V),D).permute(0,2,1,3)
    residual=query;a=layer.layer_norm2(query);query=residual+layer.linear2(torch.nn.functional.gelu(layer.linear1(a)))
    residual=query;a=layer.layer_norm3(query).reshape(len(V),F,D);a=layer.attention2(a,a,a);query=residual+a.reshape(1,len(V),F,D)
    residual=query;a=layer.layer_norm4(query);query=residual+layer.linear4(torch.nn.functional.gelu(layer.linear3(a)))
   out=m.final_layer(m.final_layer_norm(query))[0,:,0,:2].float().softmax(-1)[:,1]
  assert out.shape==(len(V),) and torch.isfinite(out).all()
  return out.cpu().numpy().astype('float64')

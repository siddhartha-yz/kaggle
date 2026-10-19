"""Proper autoregressive joint classifier; query API never accepts satisfaction labels."""
import inspect
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
CATS=['Gender','Customer Type','Type of Travel','Class']
NUMS=['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
RAW=NUMS[:2]+RATINGS+NUMS[2:]+CATS
class Encoder:
 def fit(self,X):
  assert set(X)==set(RAW) and len(X.columns)==21
  self.vocab=[sorted(X[c].dropna().astype(str).unique().tolist()) for c in CATS]
  n=self.numeric(X);self.median=np.nanmedian(n,axis=0);filled=np.where(np.isfinite(n),n,self.median);self.mean=filled.mean(0);self.scale=filled.std(0)+1e-6;self.fit_rows=len(X);return self
 def numeric(self,X):
  n=X[NUMS].to_numpy('float64',copy=True);assert np.all((~np.isfinite(n))|(n>=0));return np.log1p(n)
 def transform(self,X):
  assert set(X)==set(RAW) and len(X.columns)==21
  c=np.stack([X[name].map({v:k+1 for k,v in enumerate(vocab)}).fillna(0).to_numpy('int64') for name,vocab in zip(CATS,self.vocab)],axis=1)
  n=self.numeric(X);missing=~np.isfinite(n);n=np.clip((np.where(missing,self.median,n)-self.mean)/self.scale,-10,10);n=np.concatenate([n,missing],axis=1).astype('float32');ratings=X[RATINGS].to_numpy('float64');assert np.isfinite(ratings).all() and ((ratings>=0)&(ratings<=5)&(ratings==np.floor(ratings))).all();return c,n,ratings.astype('int64')
class JointDensity(nn.Module):
 def __init__(self,cat_sizes,number_ratings=13,number_values=6,width=64,layers=2,dropout=.05):
  super().__init__();self.number_ratings=number_ratings;self.number_values=number_values
  self.categories=nn.ModuleList([nn.Embedding(v+1,8) for v in cat_sizes]);self.metadata=nn.Sequential(nn.Linear(8*len(cat_sizes)+8,128),nn.ReLU(),nn.Linear(128,width),nn.LayerNorm(width));self.prior=nn.Linear(width,1);self.condition=nn.Embedding(2,width);self.rating=nn.Embedding(number_values+1,width);self.position=nn.Embedding(number_ratings,width)
  block=nn.TransformerEncoderLayer(width,4,width*2,dropout,batch_first=True,norm_first=True,activation='gelu');self.transformer=nn.TransformerEncoder(block,layers,enable_nested_tensor=False);self.output=nn.Linear(width,number_values)
  self.register_buffer('causal_mask',torch.triu(torch.ones(number_ratings,number_ratings,dtype=torch.bool),diagonal=1),persistent=False)
 def density_logits(self,c,n,ratings,satisfaction_condition):
  m=self.metadata(torch.cat([e(c[:,k]) for k,e in enumerate(self.categories)]+[n],dim=1));prior=self.prior(m).squeeze(1);start=torch.full((len(ratings),1),self.number_values,dtype=torch.long,device=ratings.device);previous=torch.cat([start,ratings[:,:-1]],dim=1);tokens=self.rating(previous)+self.position(torch.arange(self.number_ratings,device=ratings.device))[None]+m[:,None]+self.condition(satisfaction_condition)[:,None];h=self.transformer(tokens,mask=self.causal_mask);return prior,self.output(h)
 def joint_loss(self,c,n,ratings,target):
  prior,conditional=self.density_logits(c,n,ratings,target);likelihood=F.cross_entropy(conditional.reshape(-1,self.number_values),ratings.reshape(-1),reduction='none').reshape(len(ratings),self.number_ratings).sum(1);return (F.binary_cross_entropy_with_logits(prior,target.float(),reduction='none')+likelihood).mean()
 def log_joints(self,c,n,ratings):
  outputs=[]
  for value in [0,1]:
   condition=torch.full((len(ratings),),value,dtype=torch.long,device=ratings.device);prior,conditional=self.density_logits(c,n,ratings,condition);lp=F.logsigmoid(prior if value else -prior);outputs.append(lp+F.log_softmax(conditional,dim=-1).gather(-1,ratings[:,:,None]).squeeze(-1).sum(1))
  return torch.stack(outputs,dim=1)
 def predict_proba(self,c,n,ratings):
  joint=self.log_joints(c,n,ratings);return torch.sigmoid(joint[:,1]-joint[:,0])

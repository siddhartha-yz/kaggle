"""Outer-training-only source-target response conversion; never changes competition labels."""
import numpy as np,pandas as pd
class ResponseConversion:
 @staticmethod
 def key(raw):return raw['Class'].fillna('__MISSING__').astype(str)+'\x1f'+raw['Type of Travel'].fillna('__MISSING__').astype(str)
 def fit(self,raw,y,teacher):
  y=np.asarray(y,dtype='float64');p=np.asarray(teacher);assert len(raw)==len(y)==len(p);lo=p<.05;hi=p>.95;assert lo.sum()>=500 and hi.sum()>=500;self.alpha=float(np.clip(y[lo].mean(),.001,.45));self.beta=float(np.clip((1-y[hi]).mean(),.001,.45));self.alphas={};self.betas={};keys=self.key(raw).to_numpy();self.rows=[];self.n_train=len(y)
  for g in sorted(np.unique(keys)):
   a=lo&(keys==g);b=hi&(keys==g);alpha=float(np.clip((y[a].sum()+200*self.alpha)/(a.sum()+200),.001,.45));beta=float(np.clip(((1-y[b]).sum()+200*self.beta)/(b.sum()+200),.001,.45));self.alphas[g]=alpha;self.betas[g]=beta;self.rows.append({'group':g,'low_anchor_rows':int(a.sum()),'high_anchor_rows':int(b.sum()),'alpha':alpha,'beta':beta})
  return self
 def transform(self,raw,source_y):
  k=self.key(raw);a=k.map(self.alphas).fillna(self.alpha).to_numpy();b=k.map(self.betas).fillna(self.beta).to_numpy();assert np.all(a+b<1);q=a+(1-a-b)*np.asarray(source_y,dtype='float64');assert np.isfinite(q).all() and ((q>0)&(q<1)).all();return q
 def summary(self):return {'synthetic_training_rows':self.n_train,'global_alpha':self.alpha,'global_beta':self.beta,'groups':self.rows,'outer_validation_used':False}

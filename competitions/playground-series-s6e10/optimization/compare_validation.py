"""Paired AUC comparison; normal CIs are conditional on already-trained models.
They do not include hyperparameter search/early stopping uncertainty.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parent

def auc_influence(y,p):
 pos=p[y==1];neg=p[y==0];sp=np.sort(pos);sn=np.sort(neg)
 vp=(np.searchsorted(sn,pos,'left')+np.searchsorted(sn,pos,'right'))/(2*len(neg))
 vn=1-(np.searchsorted(sp,neg,'left')+np.searchsorted(sp,neg,'right'))/(2*len(pos))
 return vp,vn

def compare(y,base,candidate):
 a,b=auc_influence(y,base);c,d=auc_influence(y,candidate)
 delta=float(roc_auc_score(y,candidate)-roc_auc_score(y,base))
 se=float(np.sqrt(np.var(c-a,ddof=1)/len(a)+np.var(d-b,ddof=1)/len(b)))
 return dict(baseline_auc=float(roc_auc_score(y,base)),candidate_auc=float(roc_auc_score(y,candidate)),delta_auc=delta,paired_se=se,conditional_95pct_ci=[delta-1.96*se,delta+1.96*se],candidate_logloss=float(log_loss(y,candidate)))

def main():
 p=argparse.ArgumentParser();p.add_argument('candidates',nargs='+');p.add_argument('--baseline',default='artifacts/validation_predictions.csv');p.add_argument('--out',required=True)
 args=p.parse_args();bp=ROOT/args.baseline
 if bp.suffix=='.csv':
  b=pd.read_csv(bp);y=b.target.to_numpy();base=b.probability_true.to_numpy();ids=b.id.to_numpy()
 else:
  b=np.load(bp);y=b['target'];base=b['pred'];ids=b['id']
 preds={};report={}
 for name in args.candidates:
  file=ROOT/name;v=np.load(file)
  assert np.array_equal(v['id'],ids) and np.array_equal(v['target'],y)
  pred=v['pred'];preds[name]=pred;report[name]=compare(y,base,pred)
  report[name]['five_id_modulo_slice_delta_auc']=[float(roc_auc_score(y[ids%5==i],pred[ids%5==i])-roc_auc_score(y[ids%5==i],base[ids%5==i])) for i in range(5)]
 if len(preds)==2:
  a,b=preds.values()
  for w in [.25,.5,.75]:report[f'blend_first_weight_{w}']=compare(y,base,w*a+(1-w)*b)
 (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

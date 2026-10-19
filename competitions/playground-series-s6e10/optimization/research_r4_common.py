"""Continuing R4 resource/validation protocol, superseding historical time windows."""
from research_w3_common import ROOT,now,dump,sha,load_split
import os,json,resource
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.stats import t
from compare_validation import compare

def guard():
 p=json.loads((ROOT/'research_r4_plan.json').read_text());assert p['limits']['cpu_threads']==6 and not p['limits']['paid_resources']
 available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
 assert available>=2*1024**3,'Available system RAM below2GiB'
 return available

def evidence(y,base,pred,fold):
 pooled=compare(y,base,pred);byfold=[compare(y[fold==k],base[fold==k],pred[fold==k]) for k in sorted(np.unique(fold))]
 d=np.array([r['delta_auc'] for r in byfold]);K=len(d);ratio=float(np.mean([(fold==k).sum()/(fold!=k).sum() for k in sorted(np.unique(fold))]));se=float(np.sqrt((1/K+ratio)*np.var(d,ddof=1)));lower=float(d.mean()-t.ppf(.975,K-1)*se)
 return {'pooled':pooled,'fold_comparisons':byfold,'corrected_resampled_t':{'mean_delta':float(d.mean()),'se':se,'test_train_ratio':ratio,'df':K-1,'one_sided_alpha':.025,'lower_bound':lower},'eligible_for_new_partition_replication':bool(pooled['conditional_95pct_ci'][0]>0 and lower>0),'selection_bias_warning':'Reused labels; intervals conditional on frozen recipes and predictions. No untouched holdout claim.'}

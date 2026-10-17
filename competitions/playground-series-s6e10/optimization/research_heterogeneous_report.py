"""Evaluate a small, predeclared candidate set; do not optimize weights."""
from pathlib import Path
import json,numpy as np
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def main():
 plan=json.loads((ROOT/'research_heterogeneous_finalist_plan.json').read_text());b=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz');y,fold,base=b['target'],b['fold'],b['pred'];candidates={};admission={};mixtures={}
 for name,config in plan['families'].items():
  p=ROOT/config['oof_path']
  if not p.exists():admission[name]={'admitted':False,'reason':'Did not pass frozen confirmation or OOF unavailable'};continue
  a=np.load(p);assert np.array_equal(a['id'],b['id']) and np.array_equal(a['target'],y) and np.array_equal(a['fold'],fold);w=config['weight'];pred=(1-w)*base+w*a['pred'];c=compare(y,base,pred);byfold=[compare(y[fold==k],base[fold==k],pred[fold==k]) for k in range(5)];admitted=c['conditional_95pct_ci'][0]>0 and all(z['delta_auc']>0 for z in byfold);admission[name]={'admitted':admitted,'pooled':c,'byfold':byfold}
  if admitted:candidates[name]=pred;mixtures[name]={'lgb_original':1-w,name:w}
 if len(candidates)>1:
  names=list(candidates);candidates['equal_mean_admitted']=sum(candidates.values())/len(names);weights={}
  for name in names:
   for k,v in mixtures[name].items():weights[k]=weights.get(k,0)+v/len(names)
  mixtures['equal_mean_admitted']=weights
 result={};eligible=[]
 for name,pred in candidates.items():
  c=compare(y,base,pred);byfold=[compare(y[fold==k],base[fold==k],pred[fold==k]) for k in range(5)];passes=c['delta_auc']>=plan['minimum_pooled_gain'] and c['conditional_95pct_ci'][0]>0 and all(z['delta_auc']>0 for z in byfold);result[name]={'weights':mixtures[name],'pooled':c,'byfold':byfold,'eligible':passes}
  if passes:eligible.append(name)
 chosen=max(eligible,key=lambda n:result[n]['pooled']['candidate_auc']) if eligible else None
 report={'component_admission':admission,'candidates':result,'chosen':chosen,'selection_scope':'Only frozen component weights and equal-mean candidate; no public score tuning; intervals conditional and not corrected for development selection.'};(ROOT/'research_heterogeneous_oof_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

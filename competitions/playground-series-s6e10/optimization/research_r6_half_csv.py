"""Audited single externally specified half-neural candidate; no new fitting or submissions."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=json.loads((R/'research_r6_half_result.json').read_text());assert r['eligible_for_candidate'];assert sha(R/'submission_lean_realmlp_r4_cv_v7.csv')=='0ab4fe851d7677dfb1ddb0714e2c1a2f74b3f7e194d189ca6a92136a7f68c76f';oldaudit=json.loads((R/'research_r4_delivery_audit.json').read_text());assert oldaudit['status']=='passed' and oldaudit['verified_model_records']==79
 df=pd.read_csv(R/'data/train.csv');y=df.satisfaction.to_numpy('int8');errors={}
 from research_nested_stack import outer_predictions
 for stage,path in [('fivefold','r4_lean_gpu_init_oof'),('threefold','r4_lean_gpu_init_replication3')]:
  z=np.load(R/'research_artifacts'/path/'oof.npz');assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
  if stage=='fivefold':
   P=outer_predictions('oof',np.arange(len(df)),df,z['fold']);deep=np.load(R/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(deep['id'],df.id);core=np.column_stack([P,deep['pred']])
  else:
   a=np.load(R/'research_artifacts/r4_baseline_replication3/oof.npz');assert np.array_equal(a['id'],df.id) and np.array_equal(a['fold'],z['fold']);core=a['components'][:,:4]
  pred=(core@np.array([30,6,12,16])+64*z['pred'])/128;ref=(core@np.array([45,9,18,24])+32*z['pred'])/128;err=float(np.max(np.abs(ref-z['candidate'])));auc=float(roc_auc_score(y,pred));assert err<1e-12 and abs(auc-r['stages'][stage]['pooled']['candidate_auc'])<1e-14;errors[stage]={'baseline_reconstruction_error':err,'auc_recomputed':auc}
 test=pd.read_csv(R/'data/test.csv');sample=pd.read_csv(R/'data/sample_submission.csv');z=np.load(R/'research_artifacts/r4_lean_cv_final/test_components.npz');tree=np.load(R/'research_artifacts/cb_deep_full_v1/test_components.npz');assert np.array_equal(z['id'],test.id) and np.array_equal(tree['id'],test.id);pred=(30*tree['lgb_original']+6*tree['cb_original']+12*tree['residual_prior05']+16*tree['cb_deep']+64*z['realmlp'])/128;assert np.max(np.abs(pred-(.5*z['v5']+.5*z['realmlp'])))<1e-12;assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all();path=R/'submission_lean_half_r6_v8.csv'
 if not path.exists():pd.DataFrame({'id':test.id,'satisfaction':pred}).to_csv(path,index=False,float_format='%.17g')
 check=pd.read_csv(path);assert len(check)==299844 and list(check)==list(sample) and np.array_equal(check.id,test.id) and np.array_equal(check.id,sample.id);err=float(np.max(np.abs(check.satisfaction-pred)));assert err<1e-14
 out={'status':'passed','audited_utc':datetime.now(timezone.utc).isoformat(),'csv':path.name,'sha256':sha(path),'rows':len(check),'bytes':path.stat().st_size,'columns':list(check),'positive_class':True,'fixed_weights':{'v5_full':.5,'lean_cv40':.5},'expanded_weights':{'LGB_original':30/128,'CB8':6/128,'residual_prior05':12/128,'CB10':16/128,'lean_cv40':64/128},'validation':r['stages'],'independent_reconstruction':errors,'csv_roundtrip_error':err,'source_audit_inherited_sha256':sha(R/'research_r4_delivery_audit.json'),'source':'Sameofficialdata andpreviouslyauditedCC0v5core; own40NNcheckpoints; no newdata/modeldownload orunknownpredictions. Singleexternal50/50prior, no weightgrid/publicprobing.','formal_used':1,'remaining':1,'actualTop100verified':False,'recommendation':'Offlinequalifiedbackup; pointgain remainsfarbelowcurrentpublicgap. Preservefinalformalopportunitywhiletestinglargermechanisms.','selection_bias':'Allvalidationlabelsreused. Conditionalintervalsdo notcoverhistoricaladaptiveselection.'};(R/'research_r6_half_delivery_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='validation'},indent=2))
if __name__=='__main__':main()

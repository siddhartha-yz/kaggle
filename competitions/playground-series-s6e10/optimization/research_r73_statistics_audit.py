"""Independent reconstruction of all folds, ensemble means, candidate algebra and dual bounds."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import t
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((ROOT/'research_r73_plan.json').read_text());ap=json.loads((ROOT/'research_r73_audit_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r73_plan.json')
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
 frame=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);y=frame.satisfaction.to_numpy('int8');assert frame.satisfaction.dtype==bool and frame.id.is_unique and len(y)==699635;proofs={}
 for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
  out=ROOT/f'research_artifacts/r73_joint_density_{stage}';record=json.loads((out/'result.json').read_text());assert record['eligible'] and record['plan_sha256']==sha(ROOT/'research_r73_plan.json') and record['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');fold=np.full(len(y),-1,'int8')
  for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
  for a in [z,base]:assert all(np.array_equal(a[n],v) for n,v in [('id',frame.id),('target',y),('fold',fold)])
  pred=np.full(len(y),np.nan)
  for k in range(K):
   vi=np.flatnonzero(fold==k);fi=np.flatnonzero(fold!=k);run=out/f'fold{k}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['plan_sha256']==sha(ROOT/'research_r73_plan.json') and r['fit_indices_sha256']==sha(run/'fit_indices.npz') and r['encoder_sha256']==sha(run/'encoder.json') and r['prediction_sha256']==sha(run/'validation.npz') and all(sha(run/n)==h for n,h in r['member_files_sha256'].items());fit=np.load(run/'fit_indices.npz');assert np.array_equal(fit['idx'],fi) and np.array_equal(fit['id'],frame.id.iloc[fi]);q=np.load(run/'validation.npz');assert all(np.array_equal(q[n],v) for n,v in [('idx',vi),('id',frame.id.iloc[vi]),('target',y[vi])]);parts=[]
   for j in range(4):
    member=np.load(run/f'member{j}.npz');assert np.array_equal(member['idx'],vi) and np.array_equal(member['id'],frame.id.iloc[vi]) and np.isfinite(member['pred']).all() and ((member['pred']>=0)&(member['pred']<=1)).all();parts.append(member['pred'])
   pred[vi]=np.mean(np.stack(parts),axis=0);assert np.array_equal(pred[vi],q['pred'])
  a=np.clip(base['candidate'],1e-6,1-1e-6);b=np.clip(pred,1e-6,1-1e-6);numerator=a**.875*b**.125;candidate=numerator/(numerator+(1-a)**.875*(1-b)**.125);error=float(np.max(np.abs(candidate-z['candidate'])));assert error<1e-12 and np.array_equal(pred,z['pred']) and np.array_equal(z['reference_v12'],base['candidate']);paired=compare(y,base['candidate'],z['candidate']);d=np.array([roc_auc_score(y[fold==k],z['candidate'][fold==k])-roc_auc_score(y[fold==k],base['candidate'][fold==k]) for k in range(K)]);ratio=sum(float((fold==k).sum())/int((fold!=k).sum()) for k in range(K))/K;mean=float(d.mean());se=math.sqrt((1/K+ratio)*sum(float(v-mean)**2 for v in d)/(K-1));lower=mean-float(t.ppf(.975,K-1))*se;expected=record['vs_formal_v12'];assert abs(paired['candidate_auc']-expected['pooled']['candidate_auc'])<1e-14 and np.max(np.abs(np.asarray(paired['conditional_95pct_ci'])-expected['pooled']['conditional_95pct_ci']))<1e-14 and abs(lower-expected['corrected_resampled_t']['lower_bound'])<1e-14 and paired['conditional_95pct_ci'][0]>0 and lower>0;proofs[stage]=dict(pooled=paired,fold_deltas=d.tolist(),corrected_resampled_t_lower=lower,formula_error=error,eligible=True)
 result=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),main_plan_sha256=sha(ROOT/'research_r73_plan.json'),audit_plan_sha256=sha(ROOT/'research_r73_audit_plan.json'),verified_folds=8,verified_native_models=32,validation=proofs,actual_top500_verified=False);(ROOT/'research_r73_statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('STATISTICS_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()

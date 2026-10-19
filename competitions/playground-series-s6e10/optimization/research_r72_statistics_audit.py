"""Independent fold/context/query reconstruction and fixed-candidate statistical audit."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,sys
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import t
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((ROOT/'research_r72_plan.json').read_text());ap=json.loads((ROOT/'research_r72_audit_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r72_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r72_driver_state.json').read_text())['status']=='qualified_for_audit';df=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');assert len(df)==699635 and df.satisfaction.dtype==bool and df.id.is_unique;proofs={};records=[];total_chunks=0
    for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
        out=ROOT/f'research_artifacts/r72_largecontext_raw21_{stage}';main=json.loads((out/'result.json').read_text());assert main['eligible'] and main['plan_sha256']==sha(ROOT/'research_r72_plan.json') and main['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');fold=np.full(len(y),-1,'int8')
        for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
        for a in [z,base]:assert np.array_equal(a['id'],df.id) and np.array_equal(a['target'],y) and np.array_equal(a['fold'],fold)
        pred=np.full(len(y),np.nan)
        for k in range(K):
            fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);size=min(131072,len(fi)//4);order=np.random.default_rng(20261072+k+(0 if stage=='fivefold' else 10000)).permutation(fi);parts=[];assert len(np.unique(order[:4*size]))==4*size
            for j in range(4):
                run=out/f'fold{k}/context{j}';r=json.loads((run/'result.json').read_text());assert r['status']=='complete' and r['runner_sha256']==plan['source_sha256']['research_r72_worker.py'] and r['plan_sha256']==sha(ROOT/'research_r72_plan.json') and r['context_sha256']==sha(run/'context.npz') and r['prediction_sha256']==sha(run/'validation.npz')
                ctx=np.load(run/'context.npz');expected=order[j*size:(j+1)*size];assert np.array_equal(ctx['idx'],expected) and np.array_equal(ctx['id'],df.id.iloc[expected]) and not np.intersect1d(expected,vi).size
                q=np.load(run/'validation.npz');assert all(np.array_equal(q[n],v) for n,v in [('idx',vi),('id',df.id.iloc[vi]),('target',y[vi])]) and np.isfinite(q['pred']).all() and ((q['pred']>=0)&(q['pred']<=1)).all()
                chunks=0
                for start in range(0,len(vi),512):
                    take=vi[start:start+512]
                    with np.load(run/f'query{start:06d}.npz') as chunk:assert np.array_equal(chunk['idx'],take) and np.array_equal(chunk['id'],df.id.iloc[take]) and np.array_equal(chunk['pred'],q['pred'][start:start+len(take)])
                    chunks+=1
                total_chunks+=chunks;parts.append(q['pred'].copy());records.append(dict(stage=stage,fold=k,context=j,context_rows=size,query_chunks_verified=chunks,context_sha256=r['context_sha256'],validation_sha256=r['prediction_sha256']));q.close();ctx.close()
            pred[vi]=np.mean(np.stack(parts),axis=0)
        a=np.clip(base['candidate'].astype('float64'),1e-6,1-1e-6);b=np.clip(pred,1e-6,1-1e-6);numerator=a**.875*b**.125;candidate=numerator/(numerator+(1-a)**.875*(1-b)**.125);formula_error=float(np.max(np.abs(candidate-z['candidate'])));assert formula_error<1e-12 and np.max(np.abs(pred-z['pred']))<1e-15 and np.array_equal(z['reference_v12'],base['candidate']);pooled=compare(y,base['candidate'],z['candidate']);d=np.array([roc_auc_score(y[fold==k],z['candidate'][fold==k])-roc_auc_score(y[fold==k],base['candidate'][fold==k]) for k in range(K)]);ratio=sum(float((fold==k).sum())/int((fold!=k).sum()) for k in range(K))/K;mean=float(d.mean());se=math.sqrt((1/K+ratio)*sum(float(v-mean)**2 for v in d)/(K-1));lower=mean-float(t.ppf(.975,K-1))*se
        expected=main['vs_formal_v12'];assert abs(pooled['baseline_auc']-expected['pooled']['baseline_auc'])<1e-14 and abs(pooled['candidate_auc']-expected['pooled']['candidate_auc'])<1e-14 and np.max(np.abs(np.asarray(pooled['conditional_95pct_ci'])-expected['pooled']['conditional_95pct_ci']))<1e-14 and abs(lower-expected['corrected_resampled_t']['lower_bound'])<1e-14 and pooled['conditional_95pct_ci'][0]>0 and lower>0
        proofs[stage]=dict(pooled=pooled,fold_deltas=d.tolist(),corrected_resampled_t=dict(mean_delta=mean,se=se,test_train_ratio=ratio,df=K-1,one_sided_alpha=.025,lower_bound=lower),formula_error=formula_error,eligible=True)
    report=dict(status='passed',audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r72_plan.json'),audit_plan_sha256=sha(ROOT/'research_r72_audit_plan.json'),verified_contexts=len(records),verified_query_chunks=total_chunks,contexts=records,validation=proofs,actual_top500_verified=False,limitation=plan['limitations']);assert len(records)==32;(ROOT/'research_r72_statistics_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('STATISTICS_AUDIT_PASSED',total_chunks,flush=True)
if __name__=='__main__':main()

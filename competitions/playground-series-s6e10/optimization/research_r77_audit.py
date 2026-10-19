"""Independent post-gates audit: own bins/bag RNG, manual terms, rank-based CIs.

 No research prediction/statistics helper is used for these reconstructions.
 Refuses to run unless both original qualification gates have already passed.
 """
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,gc,threading,resource
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata,t
from sklearn.model_selection import StratifiedKFold
from interpret.utils._native import Native
from interpret.glassbox._ebm._ebm import construct_bins
ROOT=Path(__file__).resolve().parent
FROZEN_PLAN_SHA256='eace3a82682b9d40deb42e0daeb55aac3f9e3a49802724932d25a2679d5ac367'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);tmp=p.with_name('.'+p.name+'.audit.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def sigmoid(a):return 1./(1.+np.exp(-a))
def influence(y,s):
 positive=s[y==1];negative=s[y==0];m=len(positive);n=len(negative);r=rankdata(np.r_[positive,negative]);rp=rankdata(positive);rn=rankdata(negative)
 return (float((r[:m].sum()-m*(m+1)/2)/(m*n)),(r[:m]-rp)/n,1-(r[m:]-rn)/m)
def compare(y,base,candidate):
 a,u,v=influence(y,base);b,x,z=influence(y,candidate);se=np.sqrt(np.var(x-u,ddof=1)/len(u)+np.var(z-v,ddof=1)/len(v));return a,b,b-a,float(b-a-1.96*se)
def manual_probability(model,frame):
 # Native boundary convention is value>=cut -> next bin; NaN ->0.
 per_feature={};linear=np.full(len(frame),np.asarray(model.intercept_).reshape(-1)[0],dtype='float64')
 for term,scores in zip(model.term_features_,model.term_scores_):
  indices=[]
  for f in term:
   level=min(len(term),len(model.bins_[f]))-1;key=(f,level)
   if key not in per_feature:
    bins=model.bins_[f][level];raw=frame.iloc[:,f]
    if isinstance(bins,dict):
     code=raw.astype(str).map(bins).fillna(scores.shape[len(indices)]-1).to_numpy('int64');code[raw.isna().to_numpy()]=0
    else:
     x=raw.to_numpy('float64');code=np.searchsorted(np.asarray(bins),x,side='right')+1;code[np.isnan(x)]=0
    per_feature[key]=code
   indices.append(per_feature[key])
  linear+=np.asarray(scores)[tuple(indices)]
 return sigmoid(linear)
def equal_bins(a,b):
 if isinstance(a,dict) or isinstance(b,dict):return a==b
 return np.array_equal(a,b)

def audit():
 assert sha(ROOT/'research_r77_plan.json')==FROZEN_PLAN_SHA256
 plan=json.loads((ROOT/'research_r77_plan.json').read_text())
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r77_ebm_raw21_{stage}/result.json').read_text())['eligible']
 for name,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/name)==h,name
 frame=pd.read_csv(ROOT/'data/train.csv');assert frame.shape==(699635,23) and frame.satisfaction.dtype==bool;ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int64');raw=frame.drop(columns=['id','satisfaction']);features=list(raw);types=['nominal' if not pd.api.types.is_numeric_dtype(raw[c]) else 'continuous' for c in features];assert types.count('nominal')==4
 native=Native.get_native_singleton();assert np.array_equal(native.discretize(np.array([-1.,0.,1.,np.nan]),np.array([0.])),[1,2,2,0]);proofs=[];statistical={};begin=time.monotonic()
 for stage,K,splitseed in [('fivefold',5,20261018),('threefold',3,20261022)]:
  out=ROOT/f'research_artifacts/r77_ebm_raw21_{stage}';baseline=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');saved=np.load(out/'oof.npz');fold=np.full(len(y),-1,dtype='int8');assembled=np.full(len(y),np.nan)
  for k,(fi,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=splitseed).split(raw,y)):
   fold[vi]=k;run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());fit=json.loads((run/'training.json').read_text());assert rec['plan_sha256']==FROZEN_PLAN_SHA256 and fit['plan_sha256']==FROZEN_PLAN_SHA256 and not fit['outer_validation_used'];assert all(sha(run/n)==h for n,h in rec['files_sha256'].items());ind=np.load(run/'outer_indices.npz');assert all(np.array_equal(ind[n],v) for n,v in [('fi',fi),('vi',vi),('train_id',ids[fi]),('valid_id',ids[vi])]);v=np.load(run/'validation.npz');assert np.array_equal(v['id'],ids[vi]) and np.array_equal(v['target'],y[vi]);model=joblib.load(run/'model.joblib');assert np.array_equal(model.classes_,[0,1]) and list(model.feature_names_in_)==features and model.feature_types==types
   assert all(model.get_params()[n]==value for n,value in plan['parameters'].items()) and model.callback is None;X=raw.iloc[fi].reset_index(drop=True)
   rebuilt=construct_bins(X,y[fi],None,features,types,[1024,64],binning='quantile',min_samples_bin=1,min_unique_continuous=0,seed=42)
   assert list(rebuilt[0])==features and list(rebuilt[1])==list(model.feature_types_in_)
   for got,expected in zip(model.bins_,rebuilt[2]):assert len(got)<=len(expected) and all(equal_bins(a,b) for a,b in zip(got,expected))
   # Rebuild seed42 branch/collision logic separately, then direct native samples.
   bags=np.load(run/'inner_bags.npz');rng=native.branch_rng(native.create_rng(42));used=set()
   for j in range(8):
    while True:
     child=native.branch_rng(rng);seed=native.generate_seed(child)
     if seed not in used:break
    used.add(seed);assert np.array_equal(child,bags['rng_before'][j]);nvalid=int(np.floor(len(fi)*.15));membership=native.sample_without_replacement_stratified(child,2,len(fi)-nvalid,nvalid,y[fi]);assert np.array_equal(membership,bags['bag_matrix'][j]) and np.array_equal(child,bags['rng_after'][j]) and set(np.unique(membership))=={-1,1}
   maximum=0.
   for selected in [np.arange(min(1024,len(vi))),np.arange(max(0,len(vi)-1024),len(vi))]:
    query=raw.iloc[vi[selected]].reset_index(drop=True);manual=manual_probability(model,query);api=model.predict_proba(query)[:,1];maximum=max(maximum,float(np.max(np.abs(manual-v['pred'][selected]))),float(np.max(np.abs(manual-api))));assert maximum<1e-12
    order=np.random.default_rng(7702).permutation(len(query));assert np.max(np.abs(manual_probability(model,query.iloc[order])-manual[order]))<1e-12
   query=raw.iloc[vi[:37]].copy();query.loc[:,features[types.index('nominal')]]='AUDIT_UNSEEN_CATEGORY';assert np.max(np.abs(manual_probability(model,query)-model.predict_proba(query)[:,1]))<1e-12
   assert sha(run/'model.joblib')==fit['model_sha256'];assembled[vi]=v['pred'];proofs.append(dict(stage=stage,fold=k,model_sha256=fit['model_sha256'],own_bins_exact=True,seed42_eight_bags_exact=True,manual_first_last_probability_error=maximum));del model,rebuilt,X;gc.collect();print('AUDIT_FOLD',stage,k,maximum,flush=True)
  assert all(np.array_equal(saved[n],v) and np.array_equal(baseline[n],v) for n,v in [('id',ids),('target',y),('fold',fold)]) and np.array_equal(assembled,saved['pred']);a=np.clip(baseline['candidate'],1e-6,1-1e-6);b=np.clip(assembled,1e-6,1-1e-6);candidate=sigmoid(.9375*(np.log(a)-np.log1p(-a))+.0625*(np.log(b)-np.log1p(-b)));assert np.max(np.abs(candidate-saved['candidate']))<1e-15
  base_auc,new_auc,delta,paired_lower=compare(y,baseline['candidate'],candidate);deltas=np.array([compare(y[fold==k],baseline['candidate'][fold==k],candidate[fold==k])[2] for k in range(K)]);ratio=np.mean([(fold==k).sum()/(fold!=k).sum() for k in range(K)]);lower=float(deltas.mean()-t.ppf(.975,K-1)*np.sqrt((1/K+ratio)*deltas.var(ddof=1)));official=json.loads((out/'result.json').read_text())['vs_formal_v12'];assert abs(paired_lower-official['pooled']['conditional_95pct_ci'][0])<1e-12 and abs(lower-official['corrected_resampled_t']['lower_bound'])<1e-12 and paired_lower>0 and lower>0
  statistical[stage]=dict(baseline_auc=base_auc,candidate_auc=new_auc,delta_auc=delta,paired95_lower=paired_lower,corrected_t_lower=lower)
 result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=FROZEN_PLAN_SHA256,audit_source_sha256=sha(__file__),audit_spec_sha256=sha(ROOT/'research_r77_audit_spec.json'),fold_proofs=proofs,independent_rank_based_statistics=statistical,seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,no_new_classifier_fits=True,CSV_created=False,formal_submission=False,actual_top500_verified=False,limitations='Same official labels reused; bounds conditional on fixed recipe, do not include historical research selection.');dump(ROOT/'research_r77_independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)
def main():
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r77_ebm_raw21_{stage}/result.json').read_text())['eligible']
 stop=threading.Event()
 def monitor():
  while not stop.wait(.25):
   try:
    rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');available=int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
    if rss>8*2**30 or available<4*2**30:dump(ROOT/'research_r77_audit_resource_failure.json',dict(RSS_bytes=rss,host_available_bytes=available));os._exit(86)
   except BaseException as e:dump(ROOT/'research_r77_audit_resource_failure.json',dict(reason='telemetry failure',error=repr(e)));os._exit(86)
 threading.Thread(target=monitor,daemon=True).start()
 try:audit()
 finally:stop.set()
if __name__=='__main__':main()

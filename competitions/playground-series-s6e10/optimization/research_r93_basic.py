"""Frozen effect-size audit: missing/support/duplicates/distributions/ID segments."""
from research_r93_common import *
from scipy.stats import ks_2samp
from itertools import combinations
import resource

def counts(x):return pd.Series(np.asarray(x)).value_counts(dropna=False)
def distance(a,b,table=False):
 a=counts(a);b=counts(b);labels=sorted(set(a.index)|set(b.index));ac=a.reindex(labels,fill_value=0).to_numpy('float64');bc=b.reindex(labels,fill_value=0).to_numpy('float64');p=ac/ac.sum();q=bc/bc.sum();ps=(ac+.5)/(ac.sum()+.5*len(labels));qs=(bc+.5)/(bc.sum()+.5*len(labels))
 r=dict(TV=float(.5*np.abs(p-q).sum()),PSI=float(((ps-qs)*np.log(ps/qs)).sum()),levels=len(labels))
 if table:r['table']=[dict(value=str(l),train_count=int(c),test_count=int(d),train_fraction=float(v),test_fraction=float(w)) for l,c,d,v,w in zip(labels,ac,bc,p,q)]
 return r

def bin_spec(train):
 specs={}
 for c in CONT:
  v=train[c].dropna().to_numpy();positive_only='Delay' in c
  if positive_only:v=v[v>0]
  cuts=np.unique(np.quantile(v,np.linspace(0,1,21)[1:-1])) if len(v) else np.array([])
  specs[c]=dict(positive_only=positive_only,cuts=cuts.tolist())
 return specs

def labels(x,c,spec):
 if c not in CONT:return x[c].astype(str).to_numpy()
 s=x[c];out=np.full(len(s),MISSING,dtype=object);valid=s.notna().to_numpy()
 if spec[c]['positive_only']:
  out[valid&(s.to_numpy()==0)]='zero';out[valid&(s.to_numpy()<0)]='negative';valid=valid&(s.to_numpy()>0)
 idx=np.searchsorted(np.asarray(spec[c]['cuts']),s.to_numpy()[valid],side='left');out[valid]=np.asarray([f'bin{int(i):02}' for i in idx],dtype=object)
 return out

def support(a,b):
 return dict(train_unique=int(a.nunique(dropna=False)),test_unique=int(b.nunique(dropna=False)),test_nonmissing_unseen_rows=int((b.notna()&~b.isin(a.dropna().unique())).sum()),train_nonmissing_unseen_rows=int((a.notna()&~a.isin(b.dropna().unique())).sum()))
def key(x,cols):
 a=x[cols[0]].astype(str)
 for c in cols[1:]:a=a+'\x1f'+x[c].astype(str)
 return a.to_numpy()
def main():
 p=plan();stop,thread=guard();tick=time.monotonic()
 try:
  tr,te,tri,tei=load();spec=bin_spec(tr);tl={c:labels(tr,c,spec) for c in FEATURES};vl={c:labels(te,c,spec) for c in FEATURES};rows=[]
  for c in FEATURES:
   a=tr[c];b=te[c];row=dict(feature=c,missing_train=int(a.isna().sum()),missing_test=int(b.isna().sum()),missing_rate_train=float(a.isna().mean()),missing_rate_test=float(b.isna().mean()),support=support(a,b),binned_distribution=distance(tl[c],vl[c],True))
   row['missing_rate_abs_difference']=abs(row['missing_rate_train']-row['missing_rate_test'])
   if c in NUM:
    av=a.dropna().to_numpy();bv=b.dropna().to_numpy();sd=float(np.std(av));row.update(zero_train=int((a==0).sum()),zero_test=int((b==0).sum()),mean_train=float(av.mean()),mean_test=float(bv.mean()),standardized_mean_difference=float((bv.mean()-av.mean())/sd) if sd else None,quantile_levels=[0,.01,.05,.25,.5,.75,.95,.99,1],quantiles_train=np.quantile(av,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),quantiles_test=np.quantile(bv,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),test_below_train_min=int((bv<av.min()).sum()),test_above_train_max=int((bv>av.max()).sum()))
    if c in CONT:row['KS_statistic']=float(ks_2samp(av,bv,method='asymp').statistic)
   row['practical_flags']=dict(TV_ge_02=row['binned_distribution']['TV']>=.02,PSI_ge_01=row['binned_distribution']['PSI']>=.1,missing_abs_ge_001=row['missing_rate_abs_difference']>=.001,KS_ge_02=row.get('KS_statistic',0)>=.02)
   rows.append(row)
  print('BASIC_UNIVARIATE_COMPLETE',flush=True)
  combined=pd.concat([tr,te],ignore_index=True);h=fingerprints(combined);nt=len(tr)
  dupt=int(tr.duplicated().sum());dupe=int(te.duplicated().sum());dupc=int(combined.duplicated().sum());cross_distinct=dupc-dupt-dupe;hash_cross_rows=int(np.isin(h[nt:],np.unique(h[:nt])).sum())
  assert cross_distinct>=0
  if cross_distinct==0:assert hash_cross_rows==0,'Hashoverlapnotexactduplicate;investigatecollision'
  duplicates=dict(train_excess=dupt,test_excess=dupe,combined_excess=dupc,cross_distinct_exact_vectors=cross_distinct,test_rows_with_train_fingerprint=hash_cross_rows,ID_excluded=True)
  pairs=[['Class','Type of Travel'],['Class','Customer Type'],['Type of Travel','Customer Type'],['Inflight wifi service','Ease of Online booking'],['Food and drink','Cleanliness'],['Seat comfort','Inflight entertainment'],['Online boarding','Ease of Online booking'],['On-board service','Leg room service'],['Departure Delay in Minutes','Arrival Delay in Minutes']]
  joint=[dict(fields=cols,distribution=distance(np.char.add(np.char.add(tl[cols[0]].astype(str),'\x1f'),tl[cols[1]].astype(str)),np.char.add(np.char.add(vl[cols[0]].astype(str),'\x1f'),vl[cols[1]].astype(str)),True)) for cols in pairs]
  sample=np.random.default_rng(20261024).choice(len(tr),size=len(te),replace=False);supports=[]
  for cols in [['Flight Distance'],['Class','Type of Travel','Customer Type'],CATS,['Flight Distance','Age','Gender','Class','Customer Type','Type of Travel']]:
   a=key(tr,cols);b=key(te,cols);sa=a[sample];supports.append(dict(fields=cols,full_train_unique=int(len(np.unique(a))),full_test_unique=int(len(np.unique(b))),full_test_rows_unseen_train=int((~np.isin(b,np.unique(a))).sum()),equal_size=len(te),equal_train_rows_unseen_test=int((~np.isin(sa,np.unique(b))).sum()),equal_test_rows_unseen_train=int((~np.isin(b,np.unique(sa))).sum()),caveat='HighdimensionalunseensupportcanariseunderIID;notproofshift'))
  segments=[];segment_labels={}
  for name,x,ids,labs in [('train',tr,tri,tl),('test',te,tei,vl)]:
   order=np.argsort(ids,kind='stable');chunks=np.array_split(order,5);segment_labels[name]=chunks
   for j,ix in enumerate(chunks):segments.append(dict(source=name,segment=j,rows=len(ix),ID_min=int(ids[ix].min()),ID_max=int(ids[ix].max()),ID_diagnostic_only=True,features=[dict(feature=c,missing_rate=float(x[c].iloc[ix].isna().mean()),TV_to_source=distance(labs[c][ix],labs[c])['TV'],numeric_mean=float(x[c].iloc[ix].mean()) if c in NUM else None) for c in FEATURES]))
  last=segment_labels['train'][-1];first=segment_labels['test'][0];boundary=[dict(feature=c,distribution=distance(tl[c][last],vl[c][first],True)) for c in FEATURES]
  result=dict(status='complete_covariates_only',completed_utc=now(),plan_sha256=sha(ROOT/'research_r93_plan.json'),rows_train=len(tr),rows_test=len(te),loaded_columns=['id',*FEATURES],SAT_labels_loaded=False,bin_spec=spec,univariate=rows,prespecified_joint_distributions=joint,support_comparison=supports,duplicates=duplicates,IDs=dict(train_unique=bool(pd.Index(tri).is_unique),test_unique=bool(pd.Index(tei).is_unique),cross_overlap=int(np.intersect1d(tri,tei).size),train_range=[int(tri.min()),int(tri.max())],test_range=[int(tei.min()),int(tei.max())],note='DisjointsequentialIDrangesareexpectedbyconstruction,notdistributionevidence;notmodelinputs'),ID_segments=segments,train_last_vs_test_first=boundary,new_classifier_fits=0,seconds=time.monotonic()-tick,peak_RSS_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024)
  reserve(4);dump(OUT/'basic.json',result);print('BASIC_COMPLETE',result['seconds'],flush=True)
 finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':main()

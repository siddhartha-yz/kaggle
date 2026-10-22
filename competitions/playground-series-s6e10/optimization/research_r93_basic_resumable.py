"""Same frozen basic metrics, hashed string support and atomic per-item checkpoints."""
from research_r93_repair_common import *
from research_r93_basic import counts,distance,bin_spec,labels,support,key
from research_r93_repair_helpers import string_support
from scipy.stats import ks_2samp
import resource

def univariate(c,tr,te,tl,vl):
 a=tr[c];b=te[c];row=dict(feature=c,missing_train=int(a.isna().sum()),missing_test=int(b.isna().sum()),missing_rate_train=float(a.isna().mean()),missing_rate_test=float(b.isna().mean()),support=support(a,b),binned_distribution=distance(tl[c],vl[c],True))
 row['missing_rate_abs_difference']=abs(row['missing_rate_train']-row['missing_rate_test'])
 if c in NUM:
  av=a.dropna().to_numpy();bv=b.dropna().to_numpy();sd=float(np.std(av));row.update(zero_train=int((a==0).sum()),zero_test=int((b==0).sum()),mean_train=float(av.mean()),mean_test=float(bv.mean()),standardized_mean_difference=float((bv.mean()-av.mean())/sd) if sd else None,quantile_levels=[0,.01,.05,.25,.5,.75,.95,.99,1],quantiles_train=np.quantile(av,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),quantiles_test=np.quantile(bv,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),test_below_train_min=int((bv<av.min()).sum()),test_above_train_max=int((bv>av.max()).sum()))
  if c in CONT:row['KS_statistic']=float(ks_2samp(av,bv,method='asymp').statistic)
 row['practical_flags']=dict(TV_ge_02=row['binned_distribution']['TV']>=.02,PSI_ge_01=row['binned_distribution']['PSI']>=.1,missing_abs_ge_001=row['missing_rate_abs_difference']>=.001,KS_ge_02=row.get('KS_statistic',0)>=.02)
 return row

def main():
 p,repair=repaired_plan();stop,thread=guard();tick=time.monotonic()
 try:
  if (OUT/'basic.json').exists():
   r=json.loads((OUT/'basic.json').read_text());assert r['status']=='complete_covariates_only' and r['plan_sha256']==sha(ROOT/'research_r93_plan.json');print('REUSE_COMPLETE_BASIC',flush=True);return
  tr,te,tri,tei=load();spec=checkpoint('bin_spec',lambda:bin_spec(tr));tl={c:labels(tr,c,spec) for c in FEATURES};vl={c:labels(te,c,spec) for c in FEATURES}
  rows=[checkpoint(f'univariate_{j:02}',lambda c=c:univariate(c,tr,te,tl,vl)) for j,c in enumerate(FEATURES)]
  def duplicates():
   combined=pd.concat([tr,te],ignore_index=True);h=fingerprints(combined);nt=len(tr);dupt=int(tr.duplicated().sum());dupe=int(te.duplicated().sum());dupc=int(combined.duplicated().sum());cross=dupc-dupt-dupe;matched=int(np.isin(h[nt:],np.unique(h[:nt])).sum());assert cross>=0
   if cross==0:assert matched==0
   return dict(train_excess=dupt,test_excess=dupe,combined_excess=dupc,cross_distinct_exact_vectors=cross,test_rows_with_train_fingerprint=matched,ID_excluded=True)
  dup=checkpoint('duplicates',duplicates);joint=[]
  for j,cols in enumerate(p['basic']['joint_pairs']):
   joint.append(checkpoint(f'joint_{j:02}',lambda cols=cols:dict(fields=cols,distribution=distance(np.char.add(np.char.add(tl[cols[0]].astype(str),'\x1f'),tl[cols[1]].astype(str)),np.char.add(np.char.add(vl[cols[0]].astype(str),'\x1f'),vl[cols[1]].astype(str)),True))))
  sample=np.random.default_rng(20261024).choice(len(tr),size=len(te),replace=False);supports=[]
  for j,cols in enumerate(p['basic']['support_keys']):
   def calculate(cols=cols):return dict(fields=cols,**string_support(key(tr,cols),key(te,cols),sample),caveat='HighdimensionalunseensupportcanariseunderIID;notproofshift')
   supports.append(checkpoint(f'support_{j:02}',calculate))
  segments=[];chunks={}
  for name,x,ids,labs in [('train',tr,tri,tl),('test',te,tei,vl)]:
   chunks[name]=np.array_split(np.argsort(ids,kind='stable'),5)
   for j,ix in enumerate(chunks[name]):
    def calculate(name=name,x=x,ids=ids,labs=labs,j=j,ix=ix):return dict(source=name,segment=j,rows=len(ix),ID_min=int(ids[ix].min()),ID_max=int(ids[ix].max()),ID_diagnostic_only=True,features=[dict(feature=c,missing_rate=float(x[c].iloc[ix].isna().mean()),TV_to_source=distance(labs[c][ix],labs[c])['TV'],numeric_mean=float(x[c].iloc[ix].mean()) if c in NUM else None) for c in FEATURES])
    segments.append(checkpoint(f'IDsegment_{name}_{j}',calculate))
  last=chunks['train'][-1];first=chunks['test'][0];boundary=[checkpoint(f'boundary_{j:02}',lambda c=c:dict(feature=c,distribution=distance(tl[c][last],vl[c][first],True))) for j,c in enumerate(FEATURES)]
  result=dict(status='complete_covariates_only',completed_utc=now(),plan_sha256=sha(ROOT/'research_r93_plan.json'),repair_sha256=sha(ROOT/'research_r93_repair_manifest.json'),rows_train=len(tr),rows_test=len(te),loaded_columns=['id',*FEATURES],SAT_labels_loaded=False,bin_spec=spec,univariate=rows,prespecified_joint_distributions=joint,support_comparison=supports,duplicates=dup,IDs=dict(train_unique=bool(pd.Index(tri).is_unique),test_unique=bool(pd.Index(tei).is_unique),cross_overlap=int(np.intersect1d(tri,tei).size),train_range=[int(tri.min()),int(tri.max())],test_range=[int(tei.min()),int(tei.max())],note='DisjointsequentialIDrangesareexpectedbyconstruction,notdistributionevidence;notmodelinputs'),ID_segments=segments,train_last_vs_test_first=boundary,new_classifier_fits=0,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  reserve(4);dump(OUT/'basic.json',result);print('BASIC_COMPLETE',result['seconds'],flush=True)
 finally:stop.set();thread.join(timeout=3)
if __name__=='__main__':main()

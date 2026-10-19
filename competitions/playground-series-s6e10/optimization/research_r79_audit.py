"""Post-gates only: independent inner split, raw statistics and adapter replay."""
from research_r79_model import ROOT,configure
from research_run_deadline import check_finalization
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,gc,threading,resource,time
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from scipy.stats import rankdata,t
from tabstar.tabstar_model import TabSTARClassifier
from tabstar.tabstar_verbalizer import TabSTARVerbalizer
from tabstar.training.lora import load_finetuned
from tabstar.training.dataloader import get_dataloader

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inner_split(fi,y):
 rng=np.random.RandomState(41);singleton=np.array([rng.choice(fi[y[fi]==v]) for v in [0,1]]);rest=fi[~np.isin(fi,singleton)];n=min(int(len(fi)*.1),1000);train,valid=train_test_split(rest,test_size=n,random_state=41,stratify=y[rest]);joined=np.r_[valid,singleton];return train,joined[np.random.RandomState(41).choice(len(joined),len(joined),replace=False)]
def parameter_hash(model,base_only=False):
 h=hashlib.sha256()
 for name,tensor in sorted(model.named_parameters()):
  if base_only and any(q in name for q in ['lora_A','lora_B']):continue
  a=tensor.detach().cpu().numpy();h.update(name.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def native_probability(native,preprocessor,frame):
 data=preprocessor.transform(frame,y=None);native.eval();values=[];manual_error=0.
 for data in get_dataloader(data,is_train=False,batch_size=64):
  check_finalization()
  with torch.no_grad(),torch.autocast('cuda',enabled=True):logits=native(x_txt=data.x_txt,x_num=data.x_num,d_output=2)
  logits=logits.float();p=torch.softmax(logits,dim=1).cpu().numpy()[:,1];a=logits.cpu().numpy().astype('float64');e=np.exp(a-a.max(axis=1,keepdims=True));manual_error=max(manual_error,float(np.max(np.abs(p-e[:,1]/e.sum(axis=1)))));values.append(p)
 assert manual_error<2e-7
 return np.concatenate(values).astype('float64'),manual_error
def compare(y,a,b):
 def influence(s):
  p=s[y==1];q=s[y==0];m=len(p);n=len(q);r=rankdata(np.r_[p,q]);return (r[:m].sum()-m*(m+1)/2)/(m*n),(r[:m]-rankdata(p))/n,1-(r[m:]-rankdata(q))/m
 old,u,v=influence(a);new,x,z=influence(b);se=np.sqrt(np.var(x-u,ddof=1)/len(u)+np.var(z-v,ddof=1)/len(v));return float(old),float(new),float(new-old),float(new-old-1.96*se)
def verify_preprocessor(rebuilt,saved,raw,ti):
 assert rebuilt.y_name==saved.y_name=='satisfaction' and rebuilt.y_values==saved.y_values==['False','True'] and rebuilt.d_output==saved.d_output==2 and rebuilt.constant_columns==saved.constant_columns==[] and not rebuilt.date_transformers and not saved.date_transformers
 # Native fit iterates a set; hash-randomized dict insertion order has no
 # semantic effect because transform explicitly sorts numerical columns.
 assert set(rebuilt.numerical_transformers)==set(saved.numerical_transformers) and set(rebuilt.semantic_transformers)==set(saved.semantic_transformers)
 normalized={c.replace('_',' ').replace('-',' ').replace('.',' ').replace(':',' ') if ' ' in c else c:c for c in raw}
 for name,a in rebuilt.numerical_transformers.items():
  b=saved.numerical_transformers[name]
  for attr in ['mean_','var_','scale_','n_samples_seen_']:assert np.array_equal(getattr(a,attr),getattr(b,attr))
  x=raw.iloc[ti][normalized[name]].to_numpy('float64');assert np.allclose(a.mean_,np.nanmean(x),rtol=1e-12,atol=1e-12) and np.allclose(a.var_,np.nanvar(x),rtol=1e-12,atol=1e-10)
 for name,a in rebuilt.semantic_transformers.items():
  b=saved.semantic_transformers[name]
  for attr in ['quantiles_','references_','n_quantiles_','n_features_in_']:assert np.array_equal(getattr(a,attr),getattr(b,attr))
  x=raw.iloc[ti][normalized[name]].to_numpy('float64');q=np.nanpercentile(x,np.linspace(0,100,a.n_quantiles_));assert np.allclose(q,a.quantiles_[:,0],rtol=1e-12,atol=1e-10)
 assert np.array_equal(rebuilt.target_transformer.classes_,saved.target_transformer.classes_)
def main():
 check_finalization();plan=json.loads((ROOT/'research_r79_plan.json').read_text());ph=sha(ROOT/'research_r79_plan.json')
 assert ph=='3cbfef65149a28a7f35184b18d02b45ae56ee5c72e8c750e60621de5ead437fe'
 for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}/result.json').read_text())['eligible']
 for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
 configure();assert host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,ROOT/'research_artifacts/r79_audit_resource_failure.json'),daemon=True).start();begin=time.monotonic()
 try:
  frame=pd.read_csv(ROOT/'data/train.csv');raw=frame.drop(columns=['id','satisfaction']);ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');labels=pd.Series(np.where(y,'True','False'),name='satisfaction');proofs=[];statistics={}
  for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
   out=ROOT/f'research_artifacts/r79_tabstar_raw21_{stage}';official=json.loads((out/'result.json').read_text());assert official['plan_sha256']==ph and sha(out/'oof.npz')==official['oof_sha256'];base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');saved=np.load(out/'oof.npz');fold=np.full(len(y),-1,dtype='int8');assembled=np.full(len(y),np.nan)
   for k,(fi,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):
    check_finalization();fold[vi]=k;run=out/f'fold{k}';d=json.loads((run/'result.json').read_text());f=json.loads((run/'training.json').read_text());assert d['plan_sha256']==f['plan_sha256']==ph and all(sha(run/n)==h for n,h in d['files_sha256'].items());ti,iv=inner_split(fi,y);ix=np.load(run/'indices.npz');assert all(np.array_equal(ix[n],v) for n,v in [('fi',fi),('vi',vi),('inner_train',ti),('inner_valid',iv),('inner_train_id',ids[ti]),('inner_valid_id',ids[iv])])
    old=TabSTARClassifier.load(str(run/'model.joblib'));rebuilt=TabSTARVerbalizer(is_cls=True);rebuilt.fit(raw.iloc[ti],labels.iloc[ti]);verify_preprocessor(rebuilt,old.preprocessor_,raw,ti)
    for n,h in f['native_best_checkpoint_files'].items():assert sha(run/n)==h
    best_epoch=int(np.argmax([a['metric'] for a in f['trace']['inner_evaluations']]))+1;assert f['best_epoch']==best_epoch and len(f['trace']['epochs'])==3
    best=next((run/n).parent for n in f['native_best_checkpoint_files'] if n.endswith('adapter_model.safetensors'));native=load_finetuned(str(best),'alana89/TabSTAR');assert parameter_hash(native)==parameter_hash(old.model_)==f['final_parameters_sha256'] and parameter_hash(native,True)==plan['pretrained_base_parameters_sha256'];native.to('cuda');v=np.load(run/'validation.npz');maximum=0.;softmax_error=0.
    starts=list(range(0,len(vi),16384))+[(len(vi)//64)*64 if len(vi)%64 else len(vi)-64]
    for start in starts:
     chosen=vi[start:min(start+64,len(vi))];query=raw.iloc[chosen];a=rebuilt.transform(query,None);b=old.preprocessor_.transform(query,None);assert np.array_equal(a.x_txt,b.x_txt) and np.array_equal(a.x_num,b.x_num)
     p,error=native_probability(native,rebuilt,query);maximum=max(maximum,float(np.max(np.abs(p-v['pred'][start:start+len(chosen)]))));softmax_error=max(softmax_error,error)
    assert maximum<1e-6
    assert np.array_equal(v['idx'],vi) and np.array_equal(v['id'],ids[vi]) and np.array_equal(v['target'],y[vi]);assembled[vi]=v['pred'];proofs.append(dict(stage=stage,fold=k,inner_indices_exact=True,own_scalers_quantiles_exact=True,native_best_adapter_parameters_exact=True,prediction_error=maximum,independent_two_logit_softmax_error=softmax_error));del old,native,rebuilt;gc.collect();torch.cuda.empty_cache();print('AUDIT_FOLD',stage,k,maximum,flush=True)
   assert all(np.array_equal(saved[n],v) and np.array_equal(base[n],v) for n,v in [('id',ids),('target',y),('fold',fold)]) and np.array_equal(saved['pred'],assembled)
   a=np.clip(base['candidate'],1e-6,1-1e-6);b=np.clip(assembled,1e-6,1-1e-6);c=1/(1+np.exp(-(.9375*(np.log(a)-np.log1p(-a))+.0625*(np.log(b)-np.log1p(-b)))));assert np.max(np.abs(c-saved['candidate']))<1e-15
   oldauc,newauc,delta,lower=compare(y,base['candidate'],c);diff=np.array([compare(y[fold==k],base['candidate'][fold==k],c[fold==k])[2] for k in range(K)]);ratio=np.mean([(fold==k).sum()/(fold!=k).sum() for k in range(K)]);corrected=float(diff.mean()-t.ppf(.975,K-1)*np.sqrt((1/K+ratio)*diff.var(ddof=1)));original=json.loads((out/'result.json').read_text())['vs_formal_v12'];assert lower>0 and corrected>0 and abs(lower-original['pooled']['conditional_95pct_ci'][0])<1e-12 and abs(corrected-original['corrected_resampled_t']['lower_bound'])<1e-12;statistics[stage]=dict(baseline_auc=oldauc,candidate_auc=newauc,delta_auc=delta,paired95_lower=lower,corrected_t_lower=corrected)
  result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=ph,audit_source_sha256=sha(__file__),fold_proofs=proofs,independent_rank_statistics=statistics,seconds=time.monotonic()-begin,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_classifier_fits=0,CSV_created=False,actual_top500_verified=False)
  (ROOT/'research_r79_independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

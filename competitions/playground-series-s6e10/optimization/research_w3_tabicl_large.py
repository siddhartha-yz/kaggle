"""Frozen TabICL weights, outer-training-only routing and disjoint label contexts."""
from research_w3_tabicl_model import FrozenTabICL,config
from research_w3_common import *
import argparse,time,gc,joblib,subprocess
from threadpoolctl import threadpool_limits
from sklearn.cluster import MiniBatchKMeans
from research_w3_tabm_features import TabMFeatures
from research_external_ablation import external

def gpu_mib():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  a=line.split(',')
  if len(a)==2 and a[0].strip()==str(os.getpid()):
   try:return int(a[1].strip())
   except ValueError:pass
 return None

class Router:
 def fit(self,raw,logit):
  self.pre=TabMFeatures();n,c=self.pre.fit_transform(raw);self.mu=float(logit.mean());self.sd=max(float(logit.std()),1e-6);x=self.join(n,c,logit)
  self.km=MiniBatchKMeans(n_clusters=128,n_init=1,batch_size=8192,max_iter=100,max_no_improvement=20,random_state=20261018);self.km.fit(x);return self.km.predict(x)
 def join(self,n,c,logit):
  blocks=[n]
  for j,card in enumerate(self.pre.cardinalities):blocks.append((c[:,j,None]==np.arange(1,card)[None,:]).astype('float32'))
  blocks.append(((logit-self.mu)/self.sd).astype('float32')[:,None]);return np.concatenate(blocks,axis=1)
 def predict(self,raw,logit):
  n,c=self.pre.transform(raw);return self.km.predict(self.join(n,c,logit))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard()
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):
  assert json.loads((ROOT/f'research_artifacts/w3_tabicl_large_{prev}/result.json').read_text())['eligible_for_next'],'Earlier frozen gate failed'
 frozen=json.loads((ROOT/'research_w3_tabicl_large_plan.json').read_text());probe=json.loads((ROOT/'research_w3_tabicl_large_hardware.json').read_text());assert probe['status']=='complete' and probe['chosen']['context_rows']==4096 and probe['chosen']['n_estimators']==4
 assert sha(config(ROOT)['model_path'])==frozen['pretrained_model_sha256'];hw=setup_gpu();cap=probe['reduced_allocator_cap_bytes'];torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
 df,raw,y,splits,ref,_,fold=load_split(a.split);ext,_=external('both',df.id);p=np.clip(ext.orig_teacher_lgb.to_numpy(),1e-6,1-1e-6);logit=np.log(p/(1-p)).astype('float32');data=raw.copy();data['orig_teacher_logit']=logit
 out=ROOT/f'research_artifacts/w3_tabicl_large_{a.split}';out.mkdir(exist_ok=True);plan_hash=sha(ROOT/'research_w3_tabicl_large_plan.json');runner_hash=sha(__file__);allpred=np.full(len(y),np.nan);comps=[]
 for k,(fi,vi) in enumerate(splits):
  start_guard();run=out/f'fold{k}';run.mkdir(exist_ok=True);rec={'status':'routing','split':a.split,'fold':k,'started_utc':now(),'plan_sha256':plan_hash,'runner_sha256':runner_hash,'model_sha256':frozen['pretrained_model_sha256'],'hardware':hw,'allocator_cap_bytes':cap,'train_rows':len(fi),'valid_rows':len(vi),'completed_routes':0};stamp=run/'result.json'
  if stamp.exists():
   prev=json.loads(stamp.read_text());assert prev['plan_sha256']==plan_hash and prev['runner_sha256']==runner_hash
  dump(stamp,rec);started=time.monotonic();assert len(np.intersect1d(fi,vi))==0
  with threadpool_limits(6):
   if (run/'router.joblib').exists():router=joblib.load(run/'router.joblib');troutes=router.predict(raw.iloc[fi],logit[fi])
   else:router=Router();troutes=router.fit(raw.iloc[fi],logit[fi]);joblib.dump(router,run/'router.joblib')
   vroutes=router.predict(raw.iloc[vi],logit[vi]);np.savez_compressed(run/'routing.npz',train_idx=fi,train_route=troutes,valid_idx=vi,valid_route=vroutes);rec.update(status='in_context_prediction',router_sha256=sha(run/'router.joblib'),routing_seconds=time.monotonic()-started);dump(stamp,rec)
   model=FrozenTabICL(**config(ROOT,cache=probe['chosen']['kv_cache']));pred=np.full(len(vi),np.nan)
   for cluster in range(128):
    take=np.flatnonzero(vroutes==cluster)
    if not len(take):continue
    checkpoint=run/f'route{cluster:03d}.npz';local=fi[troutes==cluster];outside=fi[troutes!=cluster];rng=np.random.default_rng(20261018+k*1000+cluster);nl=min(2048,len(local));context=np.concatenate([rng.choice(local,nl,replace=False),rng.choice(outside,4096-nl,replace=False)]);rng.shuffle(context);assert len(np.unique(context))==4096 and np.isin(context,fi).all() and not np.isin(context,vi).any() and len(np.unique(y[context]))==2
    if checkpoint.exists():
     z=np.load(checkpoint);assert np.array_equal(z['context_idx'],context) and np.array_equal(z['idx'],vi[take]);pv=z['pred']
    else:
     gpu_epoch_guard();route_start=time.monotonic();model.fit(data.iloc[context],y[context]);assert np.array_equal(model.classes_,[0,1]);pv=np.concatenate([model.predict_proba(data.iloc[vi[take[j:j+128]]])[:,1] for j in range(0,len(take),128)]);torch.cuda.synchronize();assert np.isfinite(pv).all() and ((pv>=0)&(pv<=1)).all();np.savez_compressed(checkpoint,context_idx=context,idx=vi[take],id=df.id.iloc[vi[take]].to_numpy(),pred=pv);used=gpu_mib();assert used is None or used<=1024,'Process GPU exceeds1GiB';rec.update(last_route=cluster,last_route_seconds=time.monotonic()-route_start,process_gpu_mib=used,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    pred[take]=pv;rec.update(completed_routes=rec['completed_routes']+1,predicted_rows=int(np.isfinite(pred).sum()),last_progress_utc=now());dump(stamp,rec)
    if cluster%8==0:print(json.dumps({x:rec.get(x) for x in ['fold','completed_routes','predicted_rows','last_route_seconds','process_gpu_mib','last_progress_utc']}),flush=True)
   assert np.isfinite(pred).all();del model;gc.collect();torch.cuda.empty_cache()
  r=ref[vi] if a.split=='oof' else ref;candidate=.75*r+.25*pred;c=compare(y[vi],r,candidate);allpred[vi]=pred;comps.append(c);np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r);rec.update(status='complete',vs_v6=c,completed_utc=now(),elapsed_seconds=time.monotonic()-started);dump(stamp,rec);print(json.dumps(rec,indent=2),flush=True)
  if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps,'completed_utc':now()});return
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=.75*ref+.25*pred;c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,**({'fold':fold} if fold is not None else {}));res={'status':'complete','split':a.split,'plan_sha256':plan_hash,'runner_sha256':runner_hash,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()

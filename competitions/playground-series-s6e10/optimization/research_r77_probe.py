"""Synthetic fullshape EBM CPU/threading/RSS feasibility; require R76 terminal."""
from research_r77_model import ROOT,make_model,reconstruct_probability,parameters
import os,time,json,threading,resource,subprocess,hashlib
import numpy as np,pandas as pd,joblib
from datetime import datetime,timezone

def watch(stop,peak,path):
 while not stop.wait(.25):
  try:
   from pathlib import Path
   rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;peak[0]=max(peak[0],rss)
   children=Path(f'/proc/self/task/{os.getpid()}/children').read_text().split();reason=None
   if rss>8*2**30:reason='own RSS ceiling'
   elif available<4*2**30:reason='host available RAM reserve'
   elif children:reason='Unexpected process children; CPU threading backend required'
   if reason:path.write_text(json.dumps(dict(reason=reason,RSS_bytes=rss,host_available_bytes=available,children=children)));os._exit(86)
  except BaseException as e:path.write_text(json.dumps(dict(reason='Resource telemetry failure',error=repr(e))));os._exit(86)

def main():
 assert json.loads((ROOT/'research_r76_driver_state.json').read_text())['status'] in ['not_qualified','qualified_requires_independent_audit'],'R76 must be fully stopped before another model runs'
 available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;assert available>=6*2**30
 stop=threading.Event();peak=[0];out=ROOT/'research_artifacts/r77_cpu_probe';out.mkdir(exist_ok=True);threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start();tick=time.monotonic()
 try:
  rng=np.random.default_rng(77);n=559708;raw=pd.DataFrame({**{f'n{k}':rng.normal(size=n).astype('float32') for k in range(17)},**{f'c{k}':rng.choice(['A','B','C'],n) for k in range(4)}});y=(raw.n0.to_numpy()+raw.n1.to_numpy()+rng.normal(size=n)>.2).astype('int8');raw.loc[::29,'n0']=np.nan
  m=make_model(raw,20)
  state=dict(status='synthetic_fit_running',updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),synthetic_only=True,official_SAT_loaded=False,rows=n,features=21,outer_bags=8,max_rounds=20,CPU_threads=8,GPU_used=False)
  (ROOT/'research_r77_probe_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
  with joblib.parallel_backend('threading',n_jobs=8):
   backend=joblib.Parallel(n_jobs=8)._backend;assert type(backend).__name__=='ThreadingBackend';m.fit(raw,y)
  assert np.array_equal(m.classes_,[0,1]) and list(m.feature_names_in_)==list(raw)
  query=raw.iloc[:1024].copy();p=m.predict_proba(query)[:,1];recon=reconstruct_probability(m,query);q=query.copy();q['c0']='UNSEEN_QUERY_ONLY';unknown=m.predict_proba(q)[:,1];assert np.isfinite(unknown).all();reconstruction=float(np.max(np.abs(p-recon)));assert reconstruction<1e-12
  saved=out/'synthetic_model.joblib';joblib.dump(m,saved);loaded=joblib.load(saved);reload=float(np.max(np.abs(p-loaded.predict_proba(query)[:,1])));batch=float(np.max(np.abs(p[:37]-loaded.predict_proba(query.iloc[:37])[:,1])));assert max(reload,batch)<1e-12
  rec=dict(status='passed_no_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),synthetic_only=True,official_SAT_loaded=False,fit_rows=n,features=21,outer_bags=8,CPU_thread_backend='ThreadingBackend',GPU_used=False,max_rounds=20,terms=len(m.term_features_),seconds=time.monotonic()-tick,peak_RSS_bytes=max(peak[0],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),reload_error=reload,batch_error=batch,independent_additive_formula_error=reconstruction,unknown_query_category_finite=True,breakpoint_iteration=getattr(m,'best_iteration_',np.asarray([])).tolist(),limitations='Short synthetic fullshape capacity test only. Does not predict real full50kround early-stopped fit time or classification gain.')
  (ROOT/'research_r77_probe.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2),flush=True)
  (ROOT/'research_r77_probe_state.json').write_text(json.dumps(dict(status='passed_no_quality_score',updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid()),indent=2)+'\n')
 finally:stop.set()
if __name__=='__main__':main()

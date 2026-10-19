"""Only syntheticdata: three-epoch observation equivalence and base hash proof."""
from research_r79_model import ROOT,configure,make_model,predict,digest,trace_native,torch,np
from research_r79_probe import synthetic
from research_resource_guard_v2 import CAP,MIN_GPU_FREE,host_available,watch
from tabstar.training.lora import load_pretrained
from tabstar.training.utils import fix_seed
from tabstar.preprocessing.splits import split_to_val
import gc,time,json,threading,resource
from datetime import datetime,timezone
def main():
 configure();assert host_available()>=6*2**30 and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
 out=ROOT/'research_artifacts/r79_trace_probe';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True).start()
 try:
  fix_seed(42);initial=load_pretrained('alana89/TabSTAR',32,2,.1);basehash=digest(initial,True);del initial;gc.collect();x,y=synthetic(2048);X,I,Y,J=split_to_val(x,y,is_cls=True);prob=[];hashes=[];tr=None;times=[]
  for observed in [False,True]:
   model=make_model(out/('observed' if observed else 'original'));tick=time.monotonic()
   if observed:
    with trace_native(lambda *a,**k:None,basehash) as tr:model.fit(X,Y,x_val=I,y_val=J)
   else:model.fit(X,Y,x_val=I,y_val=J)
   times.append(time.monotonic()-tick);assert digest(model.model_,True)==basehash;prob.append(predict(model,x.iloc[:64]));hashes.append(digest(model.model_));del model;gc.collect();torch.cuda.empty_cache()
  error=float(np.max(np.abs(prob[0]-prob[1])));assert error<1e-6 and len(tr['epochs'])==3 and tr['updates_executed']>0
  d=dict(status='passed_no_official_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),synthetic_only=True,official_SAT_loaded=False,epochs=3,pretrained_base_parameters_sha256=basehash,observation_prediction_error=error,observation_parameters_exact=hashes[0]==hashes[1],fit_seconds=times,native_trace=tr,peak_GPU_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  (ROOT/'research_r79_trace_probe.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2),flush=True)
 finally:stop.set()
if __name__=='__main__':main()

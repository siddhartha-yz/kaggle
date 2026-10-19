"""Isolate execution-only numerical differences; no task quality scores."""
from research_r22_probe import ROOT,config,FrozenTabICL,torch,np,pd,json,time,gc,subprocess,resource,threading,psutil,sha,dump,CAP,CEILING
from research_r22_tabicl_offload import bounded_offload,inference_config
from tabicl._model.kv_cache import KVCacheEntry
from contextlib import nullcontext
from threadpoolctl import threadpool_limits

def snapshot(m):
    return {(name,part,k,kind):getattr(e,kind).detach().cpu().contiguous().clone() for name,c in m.model_kv_cache_.items() for part in ['col_cache','icl_cache'] for k,e in getattr(c,part).kv.items() for kind in ['key','value'] if e.is_valid()}
def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8')
    fold=np.load(ROOT/'research_artifacts/oof_v1/folds.npz')['fold'];order=np.random.default_rng(20261030).permutation(np.flatnonzero(fold!=0));A=order[:1024];query=order[-128:]
    cfg=config(ROOT,n_estimators=1,cache=True);cfg['norm_methods']=['none']
    out=ROOT/'research_artifacts/r22_parity_diagnostic_v2';out.mkdir(exist_ok=True)
    result=dict(no_quality_score=True,query_labels_used=False,results=[])
    with threadpool_limits(6):
        native=FrozenTabICL(**cfg);native.fit(raw.iloc[A],y[A]);reference=native.predict_proba(raw.iloc[query]);reference_cache=snapshot(native)
        repeat=native.predict_proba(raw.iloc[query]);result['native_repeat_error']=float(np.max(np.abs(repeat-reference)))
        result['cache_dtypes']=sorted({str(t.dtype) for t in reference_cache.values()})
        for cache in native.model_kv_cache_.values():
            for part in [cache.col_cache,cache.icl_cache]:
                for k,e in list(part.kv.items()):part.kv[k]=KVCacheEntry(e.key.contiguous(),e.value.contiguous())
        contiguous=native.predict_proba(raw.iloc[query]);result['native_contiguous_cache_error']=float(np.max(np.abs(contiguous-reference)))
        del native;gc.collect();torch.cuda.empty_cache()
        for label,cpu_cache,fixed_batches,chunk in [('cap_only_native',False,False,10**9),('fixed_native_reference',False,True,128),('fixed_cpu_cache',True,True,128)]:
            mode=bounded_offload(chunk,cpu_cache=cpu_cache,fixed_batches=fixed_batches)
            with mode as counters:
                m=FrozenTabICL(**dict(cfg,offload_mode='cpu',inference_config=inference_config()));m.fit(raw.iloc[A],y[A]);cache=snapshot(m)
                errors=[float(torch.max(torch.abs(t.float()-reference_cache[k].float()))) for k,t in cache.items()]
                p=m.predict_proba(raw.iloc[query]);delta=np.abs(p-reference)
                row=dict(label=label,max_probability_error=float(delta.max()),mean_probability_error=float(delta.mean()),max_cache_value_error=max(errors),cache_key_errors=[dict(key=str(k),error=float(torch.max(torch.abs(t.float()-reference_cache[k].float())))) for k,t in cache.items() if not torch.equal(t,reference_cache[k])],counters=dict(counters))
                if label=='fixed_native_reference':fixed_reference=p.copy();fixed_cache=cache
                if label=='fixed_cpu_cache':
                    row['vs_same_schedule_native_probability_error']=float(np.max(np.abs(p-fixed_reference)))
                    row['vs_same_schedule_native_cache_error']=max(float(torch.max(torch.abs(t.float()-fixed_cache[k].float()))) for k,t in cache.items())
                result['results'].append(row);np.savez_compressed(out/f'{label}.npz',reference=reference,pred=p,query_id=df.id.iloc[query].to_numpy())
                print('DIAGNOSTIC',row,flush=True);del m;gc.collect();torch.cuda.empty_cache()
        result['status']='complete_no_quality_score';dump(out/'result.json',result)
if __name__=='__main__':main()

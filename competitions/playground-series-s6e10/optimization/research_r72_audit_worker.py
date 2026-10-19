"""Independent own-context preprocessing and immutable-weight GPU reconstruction audit."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
from datetime import datetime,timezone
import gc,json,time,threading,hashlib,sys,resource,importlib.metadata
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from research_w3_tabicl_model import FrozenTabICL,config
from research_r22_tabicl_offload import inference_config
from research_r36_memory import streamed_ffn
from research_resource_guard_v2 import CAP,CEILING,watch,host_available,MIN_AVAILABLE,MIN_GPU_FREE
import research_r22_tabicl_offload as memory_adapter
memory_adapter.CAP=CAP
ROOT=Path(__file__).resolve().parent
now=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')

def weight_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.named_parameters()):
        digest.update(name.encode());digest.update(str((tensor.dtype,tuple(tensor.shape))).encode());flat=tensor.detach().reshape(-1)
        for start in range(0,flat.numel(),262144):digest.update(flat[start:start+262144].cpu().numpy().tobytes())
    return digest.hexdigest()

def independent_preprocessing(raw,context):
    X=raw.iloc[context];cats=list(X.select_dtypes(include=['object','string','category','boolean']));nums=list(X.select_dtypes(include='number'));assert len(cats)==4 and len(nums)==17 and len(cats)+len(nums)==21
    ordinal=OrdinalEncoder(dtype=np.int64,handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1).fit(X[cats]);imputer=SimpleImputer(keep_empty_features=True).fit(X[nums]);encoded=np.concatenate([ordinal.transform(X[cats]),imputer.transform(X[nums])],axis=1)
    keep=np.array([len(np.unique(encoded[:,i]))>1 for i in range(encoded.shape[1])]);assert keep.all();encoded=encoded[:,keep]
    mean=encoded.mean(0);scale=encoded.std(0)+1e-6;scaled=np.clip((encoded-mean)/scale,-100,100);mu=np.nanmean(scaled,axis=0);sd=np.maximum(np.nanstd(scaled,axis=0,ddof=1),1e-6);clean=scaled.copy();clean[(scaled<mu-4*sd)|(scaled>mu+4*sd)]=np.nan;mu2=np.nanmean(clean,axis=0);sd2=np.maximum(np.nanstd(clean,axis=0,ddof=1),1e-6);lower=mu2-4*sd2;upper=mu2+4*sd2
    transformed=np.maximum(-np.log1p(np.abs(scaled))+lower,scaled);transformed=np.minimum(np.log1p(np.abs(transformed))+upper,transformed)
    expected=dict(cats=cats,nums=nums,ordinal=ordinal,imputer=imputer,keep=keep,encoded_sha256=arr_sha(encoded),mean=mean,scale=scale,lower=lower,upper=upper,transformed_first=transformed[:512].copy())
    del X,encoded,scaled,clean,transformed;gc.collect();return expected

def verify_preprocessing(model,raw,context,labels,expected):
    generator=model.ensemble_generator_;encoder=model.X_encoder_.tfm_;assert model.n_samples_in_==len(context) and model.n_features_in_==21 and generator.n_estimators==1 and generator.norm_methods_==['none'] and generator.random_state==model.random_state
    assert np.array_equal(model.classes_,[0,1]) and np.array_equal(generator.y_,labels) and np.array_equal(generator.unique_filter_.features_to_keep_,expected['keep']) and arr_sha(generator.X_)==expected['encoded_sha256']
    cat=encoder.named_transformers_['categorical'];num=encoder.named_transformers_['continuous'];assert cat.unknown_value==cat.encoded_missing_value==-1 and all(np.array_equal(a,b) for a,b in zip(cat.categories_,expected['ordinal'].categories_));imp_error=float(np.max(np.abs(num.statistics_-expected['imputer'].statistics_)));assert imp_error<1e-10
    Q=raw.iloc[context[:512]];manual=np.concatenate([expected['ordinal'].transform(Q[expected['cats']]),expected['imputer'].transform(Q[expected['nums']])],axis=1);assert np.array_equal(model.X_encoder_.transform(Q),manual)
    pp=generator.preprocessors_['none'];assert pp.normalizer_ is None and pp.normalization_method=='none' and pp.outlier_threshold==4.;mean_error=float(np.max(np.abs(pp.standard_scaler_.mean_-expected['mean'])));scale_error=float(np.max(np.abs(pp.standard_scaler_.scale_-expected['scale'])));bounds_error=max(float(np.max(np.abs(pp.outlier_remover_.lower_bounds_-expected['lower']))),float(np.max(np.abs(pp.outlier_remover_.upper_bounds_-expected['upper']))));transformed_error=float(np.max(np.abs(pp.X_transformed_[:512]-expected['transformed_first'])));assert max(mean_error,scale_error,bounds_error,transformed_error)<2e-9
    return dict(context_encoding_sha256=expected['encoded_sha256'],categorical_vocabulary_sha256=hashlib.sha256(json.dumps([a.tolist() for a in cat.categories_],sort_keys=True).encode()).hexdigest(),imputation_error=imp_error,scaler_mean_error=mean_error,scaler_scale_error=scale_error,outlier_bounds_error=bounds_error,transformed_query_error=transformed_error)

def main(stage,k,j):
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True';plan=json.loads((ROOT/'research_r72_plan.json').read_text());ap=json.loads((ROOT/'research_r72_audit_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r72_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r72_driver_state.json').read_text())['status']=='qualified_for_audit'
    frame=pd.read_csv(ROOT/'data/train.csv');assert len(frame)==699635 and frame.id.is_unique and frame.satisfaction.dtype==bool;raw=frame.drop(columns=['id','satisfaction']);labels=frame.satisfaction.to_numpy('int8');assert raw.shape==(699635,21)
    K,seed=(5,20261018) if stage=='fivefold' else (3,20261022);fold=np.full(len(frame),-1,'int8')
    for a,(_,v) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(frame)),labels)):fold[v]=a
    base=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');assert np.array_equal(base['id'],frame.id) and np.array_equal(base['target'],labels) and np.array_equal(base['fold'],fold);fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);size=min(131072,len(fi)//4);order=np.random.default_rng(20261072+k+(0 if stage=='fivefold' else 10000)).permutation(fi);context=order[j*size:(j+1)*size];assert len(np.unique(order[:4*size]))==4*size and not np.intersect1d(context,vi).size
    run=ROOT/f'research_artifacts/r72_largecontext_raw21_{stage}/fold{k}/context{j}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r72_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r72_worker.py'] and rec['context_sha256']==sha(run/'context.npz') and rec['prediction_sha256']==sha(run/'validation.npz')
    assert rec['context_rows']==size and rec['feature_width']==21 and rec['norm_method']=='none' and rec['random_state']==42+j and not rec['outer_validation_used_for_fit_or_preprocessing'] and not rec['pretrained_weight_update'] and rec['repeat_error']<1e-6 and rec['peak_gpu_process_bytes']<=CEILING and rec['peak_allocated_bytes']<=CAP and rec['peak_rss_bytes']<=8589934592
    stored=np.load(run/'context.npz');assert np.array_equal(stored['idx'],context) and np.array_equal(stored['id'],frame.id.iloc[context]);saved=np.load(run/'validation.npz');assert all(np.array_equal(saved[n],v) for n,v in [('idx',vi),('id',frame.id.iloc[vi]),('target',labels[vi])]) and np.isfinite(saved['pred']).all()
    dest=ROOT/f'research_artifacts/r72_independent_audit/{stage}_fold{k}_context{j}';dest.mkdir(parents=True,exist_ok=True);receipt=dest/'result.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['status']=='passed' and old['audit_plan_sha256']==sha(ROOT/'research_r72_audit_plan.json') and old['validation_sha256']==sha(run/'validation.npz') and old['context_sha256']==sha(run/'context.npz');print('AUDIT_REUSED',stage,k,j,flush=True);return
    torch.set_num_threads(16);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE and host_available()>=MIN_AVAILABLE+2*1024**3;torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch,args=(stop,peak,dest/'resource_failure.json'),daemon=True);monitor.start();tick=time.monotonic()
    try:
        with threadpool_limits(16):expected=independent_preprocessing(raw,context)
        cfg=dict(config(ROOT,n_estimators=1,cache=True),norm_methods=['none'],random_state=42+j,offload_mode='cpu',inference_config=inference_config(),n_jobs=16)
        with threadpool_limits(16),streamed_ffn() as counters:
            model=FrozenTabICL(**cfg);model._load_model();before=weight_digest(model.model_);model.fit(raw.iloc[context],labels[context]);assert not model.model_.training;preprocessing=verify_preprocessing(model,raw,context,labels[context],expected);del expected;gc.collect()
            first=model.predict_proba(raw.iloc[vi[:512]])[:,1];last_start=((len(vi)-1)//512)*512;last=model.predict_proba(raw.iloc[vi[last_start:]])[:,1];again=model.predict_proba(raw.iloc[vi[:512]])[:,1];first_error=float(np.max(np.abs(first-saved['pred'][:512])));last_error=float(np.max(np.abs(last-saved['pred'][last_start:])));repeat=float(np.max(np.abs(first-again)));assert max(first_error,last_error,repeat)<1e-6
            after=weight_digest(model.model_);assert before==after;devices=sorted({str(e.key.device) for c in model.model_kv_cache_.values() for cache in (c.col_cache,c.icl_cache) if cache for e in cache.kv.values() if e.is_valid()});assert devices==['cpu']
            result=dict(status='passed',audited_utc=now(),stage=stage,fold=k,context=j,context_rows=size,main_plan_sha256=sha(ROOT/'research_r72_plan.json'),audit_plan_sha256=sha(ROOT/'research_r72_audit_plan.json'),context_sha256=sha(run/'context.npz'),validation_sha256=sha(run/'validation.npz'),pretrained_checkpoint_sha256=sha(ROOT/'research_sources/tabicl/tabicl-classifier-v2-20260212.ckpt'),weights_sha256_before_after=before,pretrained_weights_unchanged=True,own_context_only_preprocessing=True,preprocessing=preprocessing,first512_recreation_error=first_error,last_chunk_recreation_error=last_error,A_B_A_error=repeat,cache_devices=devices,peak_gpu_process_bytes=peak[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick,versions={n:importlib.metadata.version(n) for n in ['tabicl','torch','numpy','pandas','scikit-learn']});assert result['peak_gpu_process_bytes']<=CEILING and result['peak_allocated_bytes']<=CAP and result['peak_RSS_bytes']<=8589934592;dump(receipt,result);print('AUDIT_PASSED',stage,k,j,first_error,last_error,repeat,flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))

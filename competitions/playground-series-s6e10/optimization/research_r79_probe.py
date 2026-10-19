"""Synthetic-only, capped TabSTAR CPU/CUDA compatibility and native-fit probe.

Never loads competition labels. Refuses execution while EBM validation is active.
The classifier, preprocessing, loss, LoRA and checkpoint code remain upstream.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '6'
for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
    os.environ.pop(key, None)
os.environ.update(HF_HOME=str(ROOT/'research_artifacts/r79_hf_offline'),
                  HF_HUB_CACHE=str(ROOT/'research_artifacts/r79_hf_offline/hub'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false',
                  MPLCONFIGDIR=str(ROOT/'research_artifacts/r79_matplotlib'),
                  XDG_CACHE_HOME=str(ROOT/'research_artifacts/r79_runtime_cache'))
import gc, hashlib, json, resource, subprocess, threading, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from tabstar.tabstar_model import TabSTARClassifier
from tabstar.training.lora import load_pretrained
from research_resource_guard_v2 import CAP, MIN_GPU_FREE, host_available, watch

OUT = ROOT/'research_artifacts/r79_synthetic_probe'

def state(status, **kw):
    d = dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
             pid=os.getpid(), synthetic_only=True, official_SAT_loaded=False, **kw)
    p=ROOT/'research_r79_probe_state.json'; t=p.with_suffix('.tmp')
    t.write_text(json.dumps(d,indent=2)+'\n'); t.replace(p)
    print(json.dumps(d),flush=True)

def synthetic(n, seed=79):
    rng = np.random.default_rng(seed)
    numeric = ['Age','Flight Distance','Inflight wifi service',
               'Departure/Arrival time convenient','Ease of Online booking',
               'Gate location','Food and drink','Online boarding','Seat comfort',
               'Inflight entertainment','On-board service','Leg room service',
               'Baggage handling','Checkin service','Cleanliness',
               'Departure Delay in Minutes','Arrival Delay in Minutes']
    d={c:rng.integers(0,6,n).astype('float64') for c in numeric}
    d['Age']=rng.integers(18,81,n).astype('float64')
    d['Flight Distance']=rng.integers(50,5001,n).astype('float64')
    for c in numeric[-2:]: d[c]=np.maximum(rng.normal(15,20,n),0)
    d.update(Gender=rng.choice(['Female','Male'],n),
             **{'Customer Type':rng.choice(['Loyal Customer','disloyal Customer'],n),
                'Type of Travel':rng.choice(['Business travel','Personal Travel'],n),
                'Class':rng.choice(['Business','Eco','Eco Plus'],n)})
    x=pd.DataFrame(d)
    y=pd.Series(np.where(x['Seat comfort'].to_numpy()+rng.normal(0,2,n)>2.5,
                         'True','False'),name='satisfaction',index=x.index)
    x.loc[::29,'Arrival Delay in Minutes']=np.nan
    x.loc[::31,'Gender']=None
    return x,y

def classifier(device, **kw):
    return TabSTARClassifier(is_paper_version=True, device=device, random_state=42,
        max_epochs=1, lora_batch=64, global_batch=128, val_batch_size=64,
        output_dir=str(OUT/'native_checkpoints'), **kw)

def main():
    assert json.loads((ROOT/'research_r77_driver_state.json').read_text())['status'] in {
        'not_qualified','qualified_requires_independent_audit'}, 'EBM must be terminal'
    unit=subprocess.run(['systemctl','--user','is-active','codex-kaggle-r77-validation.service'],
                        capture_output=True,text=True,check=False)
    assert unit.returncode != 0, 'EBM controller still running'
    assert json.loads((ROOT/'research_r79_setup_state.json').read_text())['status']=='passed_dependency_and_import_only'
    receipt=json.loads((ROOT/'research_r79_download_receipt.json').read_text())
    cache=ROOT/'research_artifacts/r79_hf_offline/hub'
    for q in receipt['files']:
        p=cache/('models--'+q['repository'].replace('/','--'))/'snapshots'/q['revision']/q['filename']
        assert p.stat().st_size==q['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==q['sha256']
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and host_available()>=6*2**30
    assert torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    stop=threading.Event(); peak=[0]
    threading.Thread(target=watch,args=(stop,peak,OUT/'resource_failure.json'),daemon=True).start()
    tick=time.monotonic()
    try:
        state('synthetic_fullshape_preprocessing',rows=559708,CPU_threads=6,GPU_allocator_cap_bytes=CAP)
        large_x,large_y=synthetic(559708)
        large=classifier('cpu')
        t=time.monotonic(); td,vd=large._prepare_for_train(large_x,large_y,None,None)
        capacity=dict(rows=559708,inner_train_rows=len(td),inner_validation_rows=len(vd),
                      token_fields=td.x_txt.shape[1],seconds=time.monotonic()-t,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert len(td)+len(vd)==559708 and np.isfinite(td.x_num).all()
        del large_x,large_y,large,td,vd;gc.collect()
        x,y=synthetic(2048); query=x.iloc[:64].copy()
        state('pretrained_CPU_FP32_GPU_FP16_compatibility')
        model=classifier('cpu'); model._prepare_for_train(x,y,None,None)
        model.model_=load_pretrained('alana89/TabSTAR',lora_r=32,lora_alpha=2,dropout=.1)
        model.model_.eval();assert list(model.classes_)==['False','True']
        trainable=sum(p.numel() for p in model.model_.parameters() if p.requires_grad)
        text_trainable=sum(p.numel() for n,p in model.model_.named_parameters() if p.requires_grad and 'text_encoder' in n)
        assert trainable>0 and text_trainable==0
        model.predict_proba(query)
        t=time.monotonic();cpu=model.predict_proba(query);cpu_seconds=time.monotonic()-t
        model.model_.to('cuda');model.device=torch.device('cuda');model.use_amp=True
        model.predict_proba(query);torch.cuda.synchronize()
        timings=[]
        for _ in range(3):
            t=time.monotonic();gpu=model.predict_proba(query);torch.cuda.synchronize();timings.append(time.monotonic()-t)
        compatibility=float(np.max(np.abs(cpu-gpu)));assert compatibility<.02
        changed=query.copy();changed.loc[changed.index[1:],'Class']='Unseen Query Category'
        isolation=float(np.max(np.abs(gpu[0]-model.predict_proba(changed)[0])))
        batch=float(np.max(np.abs(gpu[:7]-model.predict_proba(query.iloc[:7]))))
        assert max(isolation,batch)<.002 and np.isfinite(gpu).all()
        del model;gc.collect();torch.cuda.empty_cache()
        state('synthetic_native_single_epoch_GPU_fit',rows=len(x))
        fitted=classifier('cuda')
        t=time.monotonic();fitted.fit(x,y);torch.cuda.synchronize();fit_seconds=time.monotonic()-t
        assert list(fitted.classes_)==['False','True']
        adapted=any(bool(torch.count_nonzero(p).item()) for n,p in fitted.model_.named_parameters() if 'lora_B' in n)
        assert adapted
        trained=fitted.predict_proba(query)
        assert np.isfinite(trained).all() and trained.shape==(len(query),2)
        fitted.model_.to('cpu');fitted.device=torch.device('cpu');fitted.use_amp=False
        saved=OUT/'own_synthetic_model.joblib';fitted.save(str(saved))
        del fitted;gc.collect();torch.cuda.empty_cache()
        loaded=TabSTARClassifier.load(str(saved));loaded.model_.to('cuda');loaded.device=torch.device('cuda');loaded.use_amp=True
        restored=loaded.predict_proba(query)
        reload_error=float(np.max(np.abs(trained-restored)));assert reload_error<1e-6
        result=dict(status='passed_no_official_quality_score',checked_utc=datetime.now(timezone.utc).isoformat(),
            synthetic_only=True,official_SAT_loaded=False,model_version='TabSTAR original 2025 checkpoint',
            source_package='tabstar==1.2.2',preprocessing_capacity=capacity,
            trainable_adapter_parameters=trainable,text_encoder_trainable_parameters=text_trainable,
            CPU_FP32_GPU_FP16_max_error=compatibility,query_isolation_error=isolation,query_batch_error=batch,
            CPU_inference_rows=64,CPU_inference_seconds=cpu_seconds,GPU_inference_seconds=timings,
            synthetic_2048row_native_fit_seconds=fit_seconds,reload_error=reload_error,
            native_checkpoint_saved=True,peak_GPU_process_bytes=peak[0],
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            GPU_allocator_cap_bytes=CAP,GPU_process_ceiling_bytes=3758096384,
            CPU_threads=6,host_RAM_reserve_bytes=4*2**30,GPU_free_reserve_bytes=MIN_GPU_FREE,
            seconds=time.monotonic()-tick,
            limitations='Synthetic capacity/timing only, no formal candidate/config/weight selection and no leaderboard claim.')
        (ROOT/'research_r79_probe.json').write_text(json.dumps(result,indent=2)+'\n')
        state('passed_no_official_quality_score',report='research_r79_probe.json')
    finally:stop.set()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('failed',error_type=type(error).__name__,error=str(error)[:500])
        raise

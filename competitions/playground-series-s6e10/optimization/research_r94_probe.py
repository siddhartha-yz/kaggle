"""Bounded resource measurement on own training rows; no SAT labels or quality scores."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='6'
import copy,hashlib,json,pathlib,resource,shutil,threading,time
from research_r94_model import *
from research_resource_guard_v2 import watch,host_available,CAP,MIN_GPU_FREE
from research_resume_policy_20261021 import check_training

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'research_artifacts/r94_rtd_resource_probe'

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(path)

def main():
    started=time.monotonic();check_training()
    plan=json.loads((ROOT/'research_r94_design.json').read_text())
    assert plan['status']=='frozen_design_and_probe_only_no_full_training_authorized'
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert not OUT.exists(),'Do not repeat completed/uncertain probe automatically'
    assert shutil.disk_usage(ROOT).free>=2*2**30+16*2**20
    assert host_available()>=6*2**30 and torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE
    assert torch.cuda.is_bf16_supported()
    OUT.mkdir(parents=True);stop=threading.Event();peak=[0]
    threading.Thread(target=watch,args=(stop,peak,OUT/'resource_failure.json'),daemon=True).start()
    def disk_guard():
        while not stop.wait(.5):
            try:
                check_training();assert time.monotonic()-started<170
                assert shutil.disk_usage(ROOT).free>=2*2**30
                assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=16*2**20
            except BaseException as error:
                dump(OUT/'resource_failure.json',dict(error=repr(error)));os._exit(86)
    threading.Thread(target=disk_guard,daemon=True).start()
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(plan['initialization_seed']);np.random.seed(plan['subset_seed'])
        path=ROOT/plan['existing_indices'];assert sha(path)==plan['existing_indices_sha256']
        z=np.load(path);ti=z['inner_train'];vi=z['vi'];assert not np.intersect1d(ti,vi).size
        chosen=np.sort(np.random.default_rng(plan['subset_seed']).choice(ti,8192,replace=False))
        assert not np.intersect1d(chosen,vi).size
        frame=pd.read_csv(ROOT/'data/train.csv',usecols=FEATURES)
        assert 'satisfaction' not in frame and 'id' not in frame
        frame=frame.iloc[chosen].copy();pre=Preprocessor().fit(frame);n,c=pre.transform(frame)
        dump(OUT/'preprocessing.json',pre.state())
        n=n.to('cuda');c=c.to('cuda');model=RTDModel(pre).cuda();initial=copy.deepcopy(model.state_dict())
        parameter_count=sum(p.numel() for p in model.parameters())
        generator=torch.Generator(device='cuda').manual_seed(plan['subset_seed'])
        records=[]
        for mode in ['RTD','classification_synthetic_targets']:
            model.load_state_dict(initial);model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            durations=[];positive=[];losses=[]
            for step in range(72):
                check_training();torch.cuda.synchronize();tick=time.monotonic()
                ix=torch.randint(len(n),(512,),generator=generator,device='cuda')
                a,b=n[ix],c[ix]
                if mode=='RTD':
                    selected=torch.rand((512,21),generator=generator,device='cuda')<0.15
                    donors=torch.randint(len(n),(512,21),generator=generator,device='cuda')
                    a,b,target=corrupt(a,b,n,c,selected,donors)
                    positive.append(float(target.mean()))
                else:
                    target=torch.randint(2,(512,),generator=generator,device='cuda').float()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    r,s=model(a,b);logits=r if mode=='RTD' else s
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
                assert torch.isfinite(loss);loss.backward()
                assert torch.isfinite(model.numeric_weight.grad).all()
                assert model.encoder.layers[0].linear1.weight.grad.abs().sum()>0
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                if step>=8:durations.append(elapsed);losses.append(float(loss.detach()))
            rec=dict(mode=mode,warmup_steps=8,timed_steps=64,batch_rows=512,
                median_step_seconds=float(np.median(durations)),mean_step_seconds=float(np.mean(durations)),
                p90_step_seconds=float(np.quantile(durations,.9)),last_synthetic_or_RTD_loss=losses[-1],
                actual_changed_fraction=float(np.mean(positive)) if positive else None,
                official_SAT_targets_used=False)
            records.append(rec);dump(OUT/(mode+'.json'),rec);print(json.dumps(rec),flush=True)
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            ref=model(n[:256],c[:256])[1].float().cpu()
        checkpoint=OUT/'probe_weights.pt';torch.save({k:v.cpu() for k,v in model.state_dict().items()},checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            again=model(n[:256],c[:256])[1].float().cpu()
            torch.cuda.synchronize();tick=time.monotonic()
            for _ in range(32):model(n[:512],c[:512])
            torch.cuda.synchronize();inference_per_512=(time.monotonic()-tick)/32
        replay=float((ref-again).abs().max());assert replay==0.0
        # Numerical forward probes are not validation quality or selected checkpoints.
        result=dict(status='passed_resource_probe_no_SAT_or_outer_quality',
            plan_sha256=sha(ROOT/'research_r94_design.json'),PID=os.getpid(),
            subset_rows=len(chosen),subset_index_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),
            subset_disjoint_from_outer_valid=True,official_SAT_loaded=False,test_data_loaded=False,
            architecture=CONFIG,parameters=parameter_count,records=records,
            inference_seconds_per_512=inference_per_512,checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=sha(checkpoint),save_reload_max_error=replay,
            peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),peak_process_GPU_bytes=peak[0],
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            total_seconds=time.monotonic()-started,disk_free_bytes=shutil.disk_usage(ROOT).free,
            full_pair_training_started=False)
        dump(ROOT/'research_r94_probe_result.json',result);print(json.dumps(result),flush=True)
    finally:stop.set()

if __name__=='__main__':main()

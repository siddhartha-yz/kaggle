"""Fixed-epoch, capped CUDA ensemble with CPU backing data; no interim scores."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc, json, sys, time, threading, subprocess, resource
import cloudpickle, joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT, now, dump, sha
from research_r7_loader import STAGE, load_split
from research_r8_features import AppendedResponseFeatures
from research_r4_lean_features import LeanFeatures
from research_r8_streaming import StreamingTabNNModule, streaming_backend
from research_r8_probe import BoundedRealMLP, CAP, CEILING

def main():
    variant = sys.argv[1]
    assert variant in ('lean', 'append')
    plan = json.loads((ROOT/'research_r8_plan.json').read_text())
    assert plan['allocator_cap_bytes'] == CAP and plan['gpu_process_ceiling_bytes'] == CEILING
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert json.loads((ROOT/'research_r8_full_probe.json').read_text())['status'] == 'passed_no_quality_score'
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0] >= CAP + int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/'research_artifacts'/f'r8_stream_{variant}_{STAGE}'
    out.mkdir(exist_ok=True)
    frozen = dict(plan_sha256=sha(ROOT/'research_r8_plan.json'), variant=variant, stage=STAGE)
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == frozen
    else:
        dump(out/'plan.json', frozen)
    df, raw, y, splits, reference, old_nn, fold = load_split('oof')
    predall = np.full(len(y), np.nan)
    peak = [0]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.5):
            q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
            for line in q.stdout.splitlines():
                try:
                    pid, memory = [int(v.strip()) for v in line.split(',')]
                except ValueError:
                    continue
                if pid == os.getpid():
                    peak[0] = max(peak[0], memory*1024**2)
                    if peak[0] > CEILING:
                        dump(out/'resource_failure.json', dict(ended_utc=now(), peak_gpu_process_bytes=peak[0]))
                        os._exit(86)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        for k, (fi, vi) in enumerate(splits):
            run = out/f'fold{k}'
            run.mkdir(exist_ok=True)
            record = run/'result.json'
            if record.exists() and json.loads(record.read_text()).get('status') == 'complete':
                rec = json.loads(record.read_text())
                assert sha(run/'model.pkl') == rec['model_sha256']
                z = np.load(run/'validation.npz')
                assert np.array_equal(z['id'], df.id.iloc[vi]) and np.array_equal(z['target'], y[vi])
                predall[vi] = z['pred']
                continue
            available = int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
            assert available >= 2*1024**3 and torch.cuda.mem_get_info()[0] >= 1024**3
            transformer = (LeanFeatures if variant == 'lean' else AppendedResponseFeatures)()
            X = transformer.fit_transform(raw.iloc[fi].reset_index(drop=True), y[fi])
            V = transformer.transform(raw.iloc[vi].reset_index(drop=True))
            assert X.shape[1] == (47 if variant == 'lean' else 60)
            joblib.dump(transformer, run/'transformer.joblib')
            config = plan['config']
            rec = dict(status='training', started_utc=now(), variant=variant, stage=STAGE,
                       fold=k, config=config, feature_width=X.shape[1], train_rows=len(fi),
                       valid_rows=len(vi), validation_used_for_fit_or_epoch_selection=False, epochs=[])
            dump(record, rec)
            dump(ROOT/'research_r8_state.json', dict(updated_utc=now(), status='training',
                 variant=variant, stage=STAGE, fold=k, pid=os.getpid(), active_training=True))
            print('START', variant, STAGE, k, flush=True)
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            previous = [tick]
            original = StreamingTabNNModule.on_train_epoch_end
            original_step = StreamingTabNNModule.training_step
            def step_hook(module, batch, batch_idx):
                if batch_idx == 0:
                    batch_devices = sorted({str(value.device) for value in batch.values()})
                    assert batch_devices == ['cuda:0']
                    rec['observed_batch_devices'] = batch_devices
                return original_step(module, batch, batch_idx)
            def epoch_hook(module):
                original(module)
                torch.cuda.synchronize()
                devices = sorted({str(p.device) for p in module.parameters()})
                assert devices == ['cuda:0'] and str(module.train_dl.ds.device) == 'cpu'
                assert rec['observed_batch_devices'] == ['cuda:0']
                assert torch.cuda.mem_get_info()[0] >= 1024**3
                moment = time.monotonic()
                rec['epochs'].append(dict(epoch=int(module.current_epoch)+1, seconds=moment-previous[0],
                    training_parameter_devices=devices, backing_data_device='cpu',
                    peak_gpu_process_bytes=peak[0], peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                previous[0] = moment
                dump(record, rec)
                print('EPOCH', k, rec['epochs'][-1], flush=True)
            StreamingTabNNModule.on_train_epoch_end = epoch_hook
            StreamingTabNNModule.training_step = step_hook
            try:
                with streaming_backend(), threadpool_limits(6):
                    model = BoundedRealMLP(**config).fit(X, y[fi])
            finally:
                StreamingTabNNModule.on_train_epoch_end = original
                StreamingTabNNModule.training_step = original_step
            assert len(rec['epochs']) == 3 and not model.cv_alg_interface_.model.creator.is_cv
            assert np.array_equal(model.classes_, [0, 1])
            pred = model.predict_proba(V)[:, 1]
            assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            model.to('cpu')
            with open(run/'model.pkl', 'wb') as handle:
                cloudpickle.dump(model, handle)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(run/'model.pkl', 'rb') as handle:
                restored = cloudpickle.load(handle)
            restored.to('cuda:0')
            again = restored.predict_proba(V.iloc[:2048])[:, 1]
            error = float(np.max(np.abs(pred[:2048]-again)))
            assert error < 1e-7
            del restored
            gc.collect()
            torch.cuda.empty_cache()
            np.savez_compressed(run/'validation.npz', id=df.id.iloc[vi].to_numpy(), target=y[vi],
                                idx=vi, pred=pred, candidate=reference[vi]+.5*(pred-old_nn[vi]))
            predall[vi] = pred
            rec.update(status='complete', ended_utc=now(), elapsed_seconds=time.monotonic()-tick,
                       model_reload_error=error, model_sha256=sha(run/'model.pkl'),
                       transformer_sha256=sha(run/'transformer.joblib'), peak_gpu_process_bytes=peak[0],
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            dump(record, rec)
            print('DONE', variant, STAGE, k, rec['elapsed_seconds'], flush=True)
            del X, V, transformer
            gc.collect()
        assert np.isfinite(predall).all()
        candidate = reference + .5*(predall-old_nn)
        np.savez_compressed(out/'oof.npz', id=df.id.to_numpy(), target=y, fold=fold, pred=predall,
                            candidate=candidate, reference_half=reference)
        dump(out/'completed.json', dict(status='complete', ended_utc=now(),
             oof_sha256=sha(out/'oof.npz'), peak_gpu_process_bytes=peak[0], no_interim_quality_scoring=True))
        dump(ROOT/'research_r8_state.json', dict(updated_utc=now(), status='stage_complete',
             variant=variant, stage=STAGE, active_training=False))
    finally:
        stop.set()
        watcher.join(timeout=5)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        dump(ROOT/'research_r8_failure.json', dict(status='failed', error=repr(error), ended_utc=now()))
        raise

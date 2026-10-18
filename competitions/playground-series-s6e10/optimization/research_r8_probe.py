"""No-quality compatibility/resource tests; no outer-validation AUC is computed."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc, json, time, threading, subprocess, resource, types
import cloudpickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from research_w3_common import ROOT, now, dump, sha
from research_r7_loader import load_split
from research_r8_features import AppendedResponseFeatures
from research_r8_streaming import StreamingTabNNModule, streaming_backend
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.training.lightning_modules import TabNNModule

CAP = 778043392
CEILING = 1073741824

class BoundedRealMLP(RealMLP_TD_Classifier):
    def get_config(self):
        config = super().get_config()
        config['init_ram_limit_gb'] = .0078125
        return config

def main():
    full = '--full' in os.sys.argv
    out = ROOT / ('research_r8_full_probe.json' if full else 'research_r8_compatibility.json')
    rec = dict(started_utc=now(), status='running', no_quality_score=True,
               allocator_cap_bytes=CAP, gpu_process_ceiling_bytes=CEILING,
               runner_sha256=sha(__file__), adapter_sha256=sha(ROOT/'research_r8_streaming.py'))
    dump(out, rec)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0] >= CAP + int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
                        rec.update(status='resource_guard_stopped', peak_gpu_process_bytes=peak[0], ended_utc=now())
                        dump(out, rec)
                        os._exit(86)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        df, raw, y, splits, *_ = load_split('oof')
        fi, vi = splits[0]
        if not full:
            fi, vi = fi[:20000], vi[:2048]
        transformer = AppendedResponseFeatures()
        X = transformer.fit_transform(raw.iloc[fi].reset_index(drop=True), y[fi])
        V = transformer.transform(raw.iloc[vi].reset_index(drop=True))
        config = json.loads((ROOT/'research_gpu_value_plan.json').read_text())['config']
        config.update(n_epochs=1, stop_epoch=1, n_ens=8 if full else 2,
                      wd=.015, lr_sched='flat_anneal', p_drop_sched='invsqrtp1e-3')
        rec.update(train_rows=len(fi), feature_width=X.shape[1], config=config, epochs=[])
        tick = time.monotonic()
        original = StreamingTabNNModule.on_train_epoch_end
        def epoch_hook(module):
            original(module)
            torch.cuda.synchronize()
            devices = sorted({str(p.device) for p in module.parameters()})
            assert devices == ['cuda:0']
            assert str(module.train_dl.ds.device) == 'cpu'
            sample = next(iter(module.train_dl))
            assert all(t.device.type == 'cuda' for t in sample.values())
            rec['epochs'].append(dict(epoch=int(module.current_epoch)+1,
                seconds=time.monotonic()-tick, training_parameter_devices=devices,
                backing_data_device=str(module.train_dl.ds.device),
                minibatch_devices=sorted({str(t.device) for t in sample.values()})))
            dump(out, rec)
        StreamingTabNNModule.on_train_epoch_end = epoch_hook
        try:
            with streaming_backend(), threadpool_limits(6):
                model = BoundedRealMLP(**config).fit(X, y[fi])
        finally:
            StreamingTabNNModule.on_train_epoch_end = original
        assert np.array_equal(model.classes_, [0, 1]) and len(rec['epochs']) == 1
        assert not model.cv_alg_interface_.model.creator.is_cv
        rec['fit_seconds'] = time.monotonic()-tick
        if not full:
            a = model.predict_proba(V)
            model.predict_proba(V.iloc[::-1])
            again = model.predict_proba(V)
            rec['a_b_a_max_error'] = float(np.max(np.abs(a-again)))
            assert rec['a_b_a_max_error'] < 1e-7
            module = model.cv_alg_interface_.model
            module.creator.static_model.to('cuda:0')
            module.get_predict_dataloader = types.MethodType(TabNNModule.get_predict_dataloader, module)
            try:
                native = model.predict_proba(V)
            finally:
                del module.get_predict_dataloader
                module.creator.static_model.to('cpu')
            rec['same_weights_native_static_max_error'] = float(np.max(np.abs(a-native)))
            assert rec['same_weights_native_static_max_error'] < 5e-6
            model.to('cpu')
            payload = cloudpickle.dumps(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            restored = cloudpickle.loads(payload)
            restored.to('cuda:0')
            b = restored.predict_proba(V)
            rec['reload_max_error'] = float(np.max(np.abs(a-b)))
            assert rec['reload_max_error'] < 1e-7
        rec.update(status='passed_no_quality_score', ended_utc=now(),
                   peak_gpu_process_bytes=peak[0],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        assert peak[0] > 0 and peak[0] <= CEILING
        dump(out, rec)
        print(json.dumps(rec, indent=2), flush=True)
    except BaseException as error:
        rec.update(status='failed', error=repr(error), ended_utc=now(), peak_gpu_process_bytes=peak[0])
        dump(out, rec)
        raise
    finally:
        stop.set()
        watcher.join(timeout=5)

if __name__ == '__main__':
    main()

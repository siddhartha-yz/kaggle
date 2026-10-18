"""Full-context unsupervised EM resource probe; no official target or score."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from datetime import datetime, timezone
import json, time, resource, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from research_original import safe_original
from research_r9_linkage_features import ExemplarLinkage
R = Path(__file__).resolve().parent
def main():
    start = time.monotonic()
    original, train, test, cols, audit = safe_original()
    z = np.load(R/'research_artifacts/oof_v1/folds.npz')
    assert np.array_equal(z['id'],train.id)
    fi = np.flatnonzero(z['fold'] != 0)
    raw, source = train.iloc[fi][cols].reset_index(drop=True), original[cols]
    with threadpool_limits(6):
        model = ExemplarLinkage().fit(raw,source)
        features = model.transform(raw.iloc[:10000],original.satisfaction.to_numpy('int8'))
    assert features.shape == (10000,7)
    rec = dict(status='passed_no_quality_score',completed_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        fit_summary=model.fit_summary,em_trace=model.trace,source_audit=audit,
        code_sha256=hashlib.sha256((R/'research_r9_linkage_features.py').read_bytes()).hexdigest(),
        no_official_target_used=True,no_quality_score=True)
    assert rec['peak_rss_bytes'] < 8*1024**3
    (R/'research_r9_probe.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec,indent=2),flush=True)
if __name__ == '__main__':main()

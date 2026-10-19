"""One nested duration algorithm, alpha.025, both admitted baselines."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,sys,numpy as np
from scipy.stats import norm,t
def compare(y,base,candidate,fold):return evidence(y,base,candidate,fold)
def check(variant,stage):
    out=ROOT/f'research_artifacts/r18_gpu_cb_{variant}_{stage}'
    assert json.loads((out/'completed.json').read_text())['status']=='complete'
    z=np.load(out/'oof.npz');oldpath='r4_lean_gpu_init_oof' if stage=='fivefold' else 'r4_lean_gpu_init_replication3'
    official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz');private=np.load(ROOT/f'research_artifacts/r14_log_pool_{stage}/oof.npz')
    assert all(np.array_equal(z[k],official[k]) and np.array_equal(z[k],private[k]) for k in ['id','target','fold'])
    assert np.max(np.abs(z['reference_v9']-private['candidate']))<1e-12
    a=compare(z['target'],private['candidate'],z['candidate'],z['fold']);b=compare(z['target'],official['candidate'],z['candidate'],z['fold'])
    result=dict(status='complete',checked_utc=now(),variant=variant,stage=stage,vs_private_v9=a,vs_official_v7=b,
        eligible=bool(a['eligible_for_new_partition_replication'] and b['eligible_for_new_partition_replication']),oof_sha256=sha(out/'oof.npz'))
    dump(out/'result.json',result);print(variant,stage,a['pooled'],result['eligible'],flush=True);return result
if __name__=='__main__':check(sys.argv[1],sys.argv[2])

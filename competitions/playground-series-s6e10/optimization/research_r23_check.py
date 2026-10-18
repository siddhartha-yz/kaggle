"""Current v10 tree + new conditional NN vs formalv10; reused-label conditional gates."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,sys,numpy as np
from research_r23_loader import pool
def check(stage):
    out=ROOT/'research_artifacts'/f'r23_conditional_lean_{stage}'
    assert json.loads((out/'completed.json').read_text())['status']=='complete'
    z=np.load(out/'oof.npz');control=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
    oldpath='r4_lean_gpu_init_oof' if stage=='fivefold' else 'r4_lean_gpu_init_replication3'
    official=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')
    assert all(np.array_equal(z[key],control[key]) and np.array_equal(z[key],official[key]) for key in ('id','target','fold'))
    assert np.max(np.abs(z['reference_v10']-control['candidate']))<1e-12
    candidate=pool(control['tree'],z['pred']);assert np.max(np.abs(candidate-z['candidate']))<1e-12
    private=evidence(z['target'],control['candidate'],candidate,z['fold'])
    direct=evidence(z['target'],official['candidate'],candidate,z['fold'])
    result=dict(status='complete',checked_utc=now(),stage=stage,vs_official_v10=private,vs_official_v7=direct,
        eligible=private['eligible_for_new_partition_replication'] and direct['eligible_for_new_partition_replication'],oof_sha256=sha(out/'oof.npz'))
    dump(out/'result.json',result);print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':check(sys.argv[1])

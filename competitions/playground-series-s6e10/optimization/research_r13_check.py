"""One predeclared uniform32 contrast against both admitted baselines."""
from research_r4_common import ROOT,now,dump,sha,evidence
import json,numpy as np,sys
def check(stage):
    out=ROOT/'research_artifacts'/f'r13_numeric_lean_{stage}'
    assert json.loads((out/'completed.json').read_text())['status']=='complete'
    z=np.load(out/'oof.npz');oldpath='r4_lean_gpu_init_oof' if stage=='fivefold' else 'r4_lean_gpu_init_replication3'
    old=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')
    assert all(np.array_equal(z[key],old[key]) for key in ('id','target','fold'))
    half=.5*(old['candidate']-.25*old['pred'])/.75+.5*old['pred']
    assert np.max(np.abs(half-z['reference_half']))<1e-12
    candidate=half+.5*(z['pred']-old['pred'])
    assert np.max(np.abs(candidate-z['candidate']))<1e-12
    control=evidence(z['target'],half,candidate,z['fold']);direct=evidence(z['target'],old['candidate'],candidate,z['fold'])
    result=dict(status='complete',checked_utc=now(),stage=stage,oof_sha256=sha(out/'oof.npz'),
        vs_private_half=control,vs_official_v7=direct,
        eligible=control['eligible_for_new_partition_replication'] and direct['eligible_for_new_partition_replication'])
    if stage=='fivefold':
        matched=np.load(ROOT/'research_artifacts/r11_uniform_lean_fivefold/oof.npz');assert np.array_equal(z['id'],matched['id'])
        result['matched_CPU_32member_diagnostic']=evidence(z['target'],matched['candidate'],candidate,z['fold'])
    dump(out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result
if __name__=='__main__':check(sys.argv[1])

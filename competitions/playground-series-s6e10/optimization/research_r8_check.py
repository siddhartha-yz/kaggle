"""Two prospective execution/view contrasts; reused labels, conditional intervals."""
import sys, json
import numpy as np
from scipy.stats import norm, t
from research_r4_common import ROOT, now, dump, evidence, sha

def contrast(y, base, candidate, fold):
    result = evidence(y, base, candidate, fold)
    # Each of two prespecified routes uses alpha .0125 in both diagnostics.
    pooled = result['pooled']
    pooled['conditional_97_5pct_ci'] = [
        float(pooled['delta_auc'] + sign*norm.ppf(.9875)*pooled['paired_se'])
        for sign in (-1, 1)]
    corrected = result['corrected_resampled_t']
    corrected['one_sided_alpha'] = .0125
    corrected['lower_bound'] = float(corrected['mean_delta']-t.ppf(.9875, corrected['df'])*corrected['se'])
    result['eligible_for_new_partition_replication'] = bool(
        pooled['conditional_97_5pct_ci'][0] > 0 and corrected['lower_bound'] > 0)
    return result

def check(variant, stage):
    out = ROOT/'research_artifacts'/f'r8_stream_{variant}_{stage}'
    assert json.loads((out/'completed.json').read_text())['status'] == 'complete'
    z = np.load(out/'oof.npz')
    old_path = 'r4_lean_gpu_init_oof' if stage == 'fivefold' else 'r4_lean_gpu_init_replication3'
    old = np.load(ROOT/'research_artifacts'/old_path/'oof.npz')
    assert all(np.array_equal(z[key], old[key]) for key in ('id','target','fold'))
    expected_half = .5*(old['candidate']-.25*old['pred'])/.75+.5*old['pred']
    assert np.max(np.abs(expected_half-z['reference_half'])) < 1e-12
    candidate = expected_half+.5*(z['pred']-old['pred'])
    assert np.max(np.abs(candidate-z['candidate'])) < 1e-12
    result = dict(checked_utc=now(), variant=variant, stage=stage,
                  oof_sha256=sha(out/'oof.npz'),
                  vs_private_half=contrast(z['target'], expected_half, candidate, z['fold']),
                  vs_official_v7=contrast(z['target'], old['candidate'], candidate, z['fold']))
    result['eligible'] = all(result[key]['eligible_for_new_partition_replication']
                             for key in ('vs_private_half','vs_official_v7'))
    dump(out/'result.json', result)
    print(variant, stage, 'delta_half', result['vs_private_half']['pooled']['delta_auc'],
          'eligible', result['eligible'], flush=True)
    return result

if __name__ == '__main__':
    check(sys.argv[1], sys.argv[2])

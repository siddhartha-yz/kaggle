"""Summarize the frozen diagnostic; never load row data, fit, or select a model."""
import hashlib
import json
import pathlib
import shutil
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'research_artifacts/r93_official_distribution_audit'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = read(ROOT / 'research_r93_result.json')
    basic = read(OUT / 'basic.json')
    av = read(OUT / 'adversarial.json')
    integrity = read(OUT / 'integrity.json')
    assert state['status'] == 'completed_diagnostic_pending_interpretation'
    assert integrity['status'] == 'passed_diagnostic_integrity'
    for name, expected in integrity['files_sha256'].items():
        assert sha(OUT / name) == expected, name
    for name in ['basic', 'adversarial', 'integrity']:
        assert sha(OUT / (name + '.json')) == state[name + '_sha256']
    plan_sha = sha(ROOT / 'research_r93_plan.json')
    repair_sha = sha(ROOT / 'research_r93_repair_manifest.json')
    assert state['plan_sha256'] == plan_sha
    assert state['repair_sha256'] == repair_sha
    repair = read(ROOT / 'research_r93_repair_manifest.json')
    for name, expected in repair['corrected_source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert basic['SAT_labels_loaded'] is False
    assert av['official_SAT_loaded'] is False
    assert av['new_domain_classifier_fits'] == 2
    assert av['new_SAT_classifier_fits'] == 0
    assert av['negative_control_passed'] is True
    assert av['material_generalizable_distinguishability'] is False
    assert integrity['saved_model_replay_max_errors'] == [0.0, 0.0]
    u = basic['univariate']
    max_tv = max(u, key=lambda x: x['binned_distribution']['TV'])
    max_psi = max(u, key=lambda x: x['binned_distribution']['PSI'])
    continuous = [x for x in u if 'KS_statistic' in x]
    max_ks = max(continuous, key=lambda x: x['KS_statistic'])
    flagged = [{'feature': x['feature'], 'flags': x['practical_flags']}
               for x in u if any(x['practical_flags'].values())]
    joints = [dict(fields=x['fields'], TV=x['distribution']['TV'],
                   PSI=x['distribution']['PSI'])
              for x in basic['prespecified_joint_distributions']]
    segments = [dict(source=s['source'], segment=s['segment'], **f)
                for s in basic['ID_segments'] for f in s['features']]
    boundary = [dict(feature=x['feature'], TV=x['distribution']['TV'],
                     PSI=x['distribution']['PSI'])
                for x in basic['train_last_vs_test_first']]
    result = dict(
        status='closed_diagnostic_no_SAT_experiment_supported',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=plan_sha, repair_sha256=repair_sha,
        result_sha256=sha(ROOT / 'research_r93_result.json'),
        evidence_sha256={name: sha(OUT / name) for name in
                         ['basic.json', 'adversarial.json', 'integrity.json']},
        initial_failure_preserved='research_r93_initial_attempt.json',
        implementation_repair=dict(
            change='Replace quadratic object-array membership with hashed canonical-string membership; incremental atomic checkpoints and reusable fit/prediction receipts.',
            synthetic_equivalence=read(ROOT / 'research_r93_membership_equivalence.json'),
            scientific_settings_and_operation_budgets_unchanged=True,
            original_failed_180_seconds_not_erased=True),
        official_covariates_only=dict(
            rows_train=basic['rows_train'], rows_test=basic['rows_test'],
            feature_count=21, SAT_labels_loaded=False,
            ID_excluded_from_domain_models=True,
            flagged_features=flagged,
            maximum_univariate_TV=dict(feature=max_tv['feature'], value=max_tv['binned_distribution']['TV']),
            maximum_univariate_PSI=dict(feature=max_psi['feature'], value=max_psi['binned_distribution']['PSI']),
            maximum_continuous_KS=dict(feature=max_ks['feature'], value=max_ks['KS_statistic']),
            missing=[{k: v for k, v in x.items() if k.startswith('missing') or k == 'feature'}
                     for x in u if x['missing_train'] or x['missing_test']],
            support_comparison=basic['support_comparison'],
            duplicates=basic['duplicates'], IDs=basic['IDs'],
            maximum_joint_TV=max(joints, key=lambda x: x['TV']),
            maximum_ID_segment_TV=max(segments, key=lambda x: x['TV_to_source']),
            maximum_boundary_TV=max(boundary, key=lambda x: x['TV'])),
        fixed_domain_models=dict(
            real_holdout=av['real_holdout'],
            permuted_fit_label_negative_control=av['permuted_fit_label_negative_control'],
            fit_class_counts=av['fit_class_counts'],
            holdout_class_counts=av['holdout_class_counts'],
            negative_control_passed=av['negative_control_passed'],
            material_generalizable_distinguishability=av['material_generalizable_distinguishability'],
            prespecified_investment_screen=av['predeclared_material_line'],
            saved_model_replay_max_errors=integrity['saved_model_replay_max_errors'],
            exactly_two_domain_fits=True, new_SAT_fits=0,
            group_permutations_not_run_because_screen_failed=True),
        decision=dict(
            next_SAT_experiment_supported=False,
            candidate='Domain probability reweighting of satisfaction training',
            reason='No material heldout domain discrimination under the prespecified model, and no practical univariate flags. Do not invent domain weights from near-random scores.',
            no_threshold_model_or_fusion_search=True,
            public_leaderboard_not_used=True,
            missing_new_information='A distinct, falsifiable mechanism or evidence unavailable to this covariate-only audit; no evidence here to justify another drift-driven SAT fit.',
            limitations=[
                'A fixed finite model and selected marginals cannot prove identical distributions or exclude weaker/higher-order covariate shift.',
                'No test satisfaction labels were loaded; P(Y|X) shift is not identified.',
                'High-dimensional unseen keys occur under finite sampling; equal-sized six-key unseen counts are nearly symmetric and do not establish shift.',
                'Domain AUC confidence intervals are conditional on the frozen split/model; they are not universal absence-of-drift bounds.',
                'Historical satisfaction validation labels have been repeatedly reused; a different fold seed would not create an untouched holdout.']),
        resource_record=dict(
            basic_seconds=basic['seconds'], domain_stage_seconds=av['seconds'],
            basic_peak_RSS_bytes=basic['peak_RSS_bytes'],
            domain_peak_RSS_bytes=av['peak_RSS_bytes'],
            artifact_bytes_at_controller_completion=state['artifact_bytes'],
            artifact_cap_bytes=32 * 1024**2,
            disk_free_bytes_at_close=shutil.disk_usage(ROOT).free,
            disk_reserve_bytes=2 * 1024**3, GPU_used=False),
        no_new_submissions=True, no_deletions=True, main_unchanged=True,
        project_top500_goal_completed=False)
    assert not flagged
    assert result['resource_record']['disk_free_bytes_at_close'] > 2 * 1024**3
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(encoded) < 32 * 1024**2
    target = ROOT / 'research_r93_conclusion.json'
    target.write_bytes(encoded)
    print(json.dumps(dict(status=result['status'], conclusion_sha256=sha(target),
                          domain_AUC=av['real_holdout']['AUC'],
                          disk_free_GiB=shutil.disk_usage(ROOT).free / 1024**3)))


if __name__ == '__main__':
    main()

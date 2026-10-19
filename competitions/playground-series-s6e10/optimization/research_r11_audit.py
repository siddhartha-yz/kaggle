"""Independent membership, encoder, ensemble and dual-reference reconstruction."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, inspect
import numpy as np
import pandas as pd
import joblib, cloudpickle, torch
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from research_r4_common import ROOT, now, dump, sha
from research_r4_common import evidence as contrast
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures

DATA_HASHES = {
 'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
 'test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6',
 'sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}

def main():
    torch.set_num_threads(6)
    finalplan = json.loads((ROOT/'research_r11_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256'] == sha(ROOT/'research_r11_plan.json')
    for name, digest in finalplan['code_sha256'].items(): assert sha(ROOT/name) == digest
    for name, digest in DATA_HASHES.items():
        assert sha(ROOT/'data'/f'{name}.csv') == digest
    plan = json.loads((ROOT/'research_r11_plan.json').read_text())
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest
    variant = 'lean'
    df = pd.read_csv(ROOT/'data/train.csv')
    assert df.satisfaction.dtype == bool and len(df) == 699635
    raw, y = df.drop(columns=['id','satisfaction']), df.satisfaction.to_numpy('int8')
    records, checks = [], {}
    for stage, K, seed, old_path in [
        ('fivefold',5,20261018,'r4_lean_gpu_init_oof'),
        ('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
        out = ROOT/'research_artifacts'/f'r11_uniform_{variant}_{stage}'
        result = json.loads((out/'result.json').read_text())
        assert result['eligible']
        z, old = np.load(out/'oof.npz'), np.load(ROOT/'research_artifacts'/old_path/'oof.npz')
        assert np.array_equal(z['id'], df.id) and np.array_equal(z['target'], y)
        assert all(np.array_equal(z[key],old[key]) for key in ('id','target','fold'))
        expected = np.full(len(y),-1,'int8')
        for k, (_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):
            expected[vi] = k
        assert np.array_equal(expected, z['fold'])
        nn = np.zeros(len(y))
        for k in range(K):
            fi, vi = np.flatnonzero(expected != k), np.flatnonzero(expected == k)
            run = out/f'fold{k}'
            rec = json.loads((run/'result.json').read_text())
            assert rec['status'] == 'complete' and rec['config'] == plan['config']
            assert rec['model_sha256'] == sha(run/'model.pkl')
            assert rec['transformer_sha256'] == sha(run/'transformer.joblib')
            assert rec['train_rows'] == len(fi) and rec['valid_rows'] == len(vi)
            assert not rec['validation_used_for_fit_or_epoch_selection']
            assert rec['observed_batch_devices'] == ['cuda:0'] and len(rec['epochs']) == 3
            assert rec['peak_gpu_process_bytes'] <= plan['gpu_process_ceiling_bytes']
            assert all(e['training_parameter_devices'] == ['cuda:0'] and e['backing_data_device'] == 'cpu'
                       and e['peak_allocated_bytes'] <= plan['allocator_cap_bytes'] for e in rec['epochs'])
            tr = joblib.load(run/'transformer.joblib')
            assert list(raw) == tr.columns and abs(tr.encoder.target_mean_-y[fi].mean()) < 1e-15
            assert tr.encoder.cv == 5 and tr.encoder.random_state == 42
            keys = pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True), cols)
                                 for j,cols in enumerate(COMBOS)})
            for j,c in enumerate(keys):
                assert set(tr.encoder.categories_[j]) == set(keys[c])
            for c,counts in tr.counts.items():
                assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
            if variant == 'append':
                assert tr.response_encoder.cv == 5 and tr.response_encoder.random_state == 20261025
                assert abs(tr.response_encoder.target_mean_-y[fi].mean()) < 1e-15
                response_keys = tr.response_keys(raw.iloc[fi].reset_index(drop=True))
                for j,c in enumerate(response_keys):
                    assert set(tr.response_encoder.categories_[j]) == set(response_keys[c])
            sample = tr.transform(raw.iloc[vi[:64]].reset_index(drop=True))
            assert sample.shape[1] == (60 if variant == 'append' else 47)
            with open(run/'model.pkl','rb') as handle:
                model = cloudpickle.load(handle)
            module = model.cv_alg_interface_.model
            assert np.array_equal(model.classes_, [0,1]) and module.config['n_ens'] == 32
            assert not module.creator.is_cv and module.progress.max_epochs == 3
            assert all(p.device.type == 'cpu' for p in module.parameters())
            v = np.load(run/'validation.npz')
            model.to('cpu')
            module.creator.device_info = 'cpu'
            replay_input = tr.transform(raw.iloc[vi[:2048]].reset_index(drop=True))
            with threadpool_limits(6): replay = model.predict_proba(replay_input)[:,1]
            replay_error = float(np.max(np.abs(replay-v['pred'][:2048])))
            assert replay_error < 2e-6
            del model, module, replay_input
            assert np.array_equal(v['id'],df.id.iloc[vi]) and np.array_equal(v['target'],y[vi])
            nn[vi] = v['pred']
            records.append(dict(stage=stage,fold=k,model_sha256=rec['model_sha256'],native_ensemble_members=32, CPU_prediction_replay_error=replay_error))
        assert np.max(np.abs(nn-z['pred'])) < 1e-12
        if stage == 'fivefold':
            from research_nested_stack import outer_predictions
            core = outer_predictions('oof',np.arange(len(df)),df,z['fold'])
            deep = np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz')
            assert np.array_equal(deep['id'],df.id)
            core = np.column_stack([core,deep['pred']])
        else:
            base = np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz')
            assert np.array_equal(base['id'],df.id) and np.array_equal(base['fold'],z['fold'])
            core = base['components'][:,:4]
        official = (core@np.array([45,9,18,24])+32*old['pred'])/128
        private = (core@np.array([30,6,12,16])+64*old['pred'])/128
        candidate = (core@np.array([30,6,12,16])+64*nn)/128
        assert np.max(np.abs(official-old['candidate'])) < 1e-12
        assert np.max(np.abs(private-z['reference_half'])) < 1e-12
        assert np.max(np.abs(candidate-z['candidate'])) < 1e-12
        checks[stage] = dict(vs_official_v7=contrast(y,official,candidate,z['fold']),
                             vs_private_half=contrast(y,private,candidate,z['fold']))
        assert all(value['eligible_for_new_partition_replication'] for value in checks[stage].values())
    result = dict(status='passed',audited_utc=now(),variant=variant,data_sha256=DATA_HASHES,
        protocol_sha256=sha(ROOT/'research_r11_plan.json'),verified_model_records=len(records),
        verified_native_neural_members=32*len(records),model_records=records,validation=checks,
        fixed_final_formula='.5*v5full+.5*mean(five_saved_native32_ensembles)',
        target_mean_counts_categories_and_fold_membership_verified=True,
        sklearn_fit_transform_source_sha256=__import__('hashlib').sha256(inspect.getsource(TargetEncoder.fit_transform).encode()).hexdigest(),
        no_outer_validation_passed_to_fit=True,actual_top100_verified=False,
        selection_bias='Reused validation labels; conditional bounds omit adaptive historical selection.')
    dump(ROOT/'research_r11_pre_inference_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_records','validation')},indent=2))

if __name__ == '__main__': main()

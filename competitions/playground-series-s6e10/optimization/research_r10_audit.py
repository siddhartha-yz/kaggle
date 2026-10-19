"""Audit nested split isolation, epoch choice, maps and independent prediction replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='6'
from research_r4_common import ROOT,now,dump,sha,evidence
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,gc,threading,subprocess,hashlib
import numpy as np,pandas as pd,torch,joblib,cloudpickle
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures
from research_r8_probe import CAP,CEILING

HASHES={'train':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
 'test':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6',
 'sample_submission':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}

def maps(tr,raw,y,idx):
    assert tr.encoder.cv==5 and tr.encoder.random_state==42
    assert abs(tr.encoder.target_mean_-y[idx].mean())<1e-15
    assert list(raw)==tr.columns and len(tr.output_columns)==47
    subset=raw.iloc[idx].reset_index(drop=True)
    for c,counts in tr.counts.items():assert counts.sort_index().equals(subset[c].value_counts().sort_index())
    keys=pd.DataFrame({f'combo{j}':DistanceFeatures._key(subset,cols) for j,cols in enumerate(COMBOS)})
    for j,c in enumerate(keys):assert set(tr.encoder.categories_[j])==set(keys[c])

def main():
    finalplan=json.loads((ROOT/'research_r10_final_inference_plan.json').read_text())
    assert finalplan['main_plan_sha256']==sha(ROOT/'research_r10_plan.json')
    for name,digest in finalplan['code_sha256'].items():assert sha(ROOT/name)==digest
    plan=json.loads((ROOT/'research_r10_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in HASHES.items():assert sha(ROOT/'data'/f'{name}.csv')==digest
    df=pd.read_csv(ROOT/'data/train.csv');assert df.satisfaction.dtype==bool and len(df)==699635
    y=df.satisfaction.to_numpy('int8');raw=df.drop(columns=['id','satisfaction'])
    assert json.loads((ROOT/'research_r4_delivery_audit.json').read_text())['status']=='passed'
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+int(1.25*1024**3)
    torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    peak=[0];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():
                    peak[0]=max(peak[0],memory*1024**2)
                    if peak[0]>CEILING:os._exit(86)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    models,checks=[],{}
    try:
        for stage,K,seed,oldpath in [('fivefold',5,20261018,'r4_lean_gpu_init_oof'),('threefold',3,20261022,'r4_lean_gpu_init_replication3')]:
            out=ROOT/'research_artifacts'/f'r10_nested_epochs_{stage}'
            assert json.loads((out/'result.json').read_text())['eligible']
            z=np.load(out/'oof.npz');old=np.load(ROOT/'research_artifacts'/oldpath/'oof.npz')
            assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y)
            fold=np.full(len(y),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
            assert np.array_equal(fold,z['fold']) and np.array_equal(fold,old['fold'])
            nn=np.full(len(y),np.nan)
            for k in range(K):
                fi,vi=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
                it,iv=train_test_split(fi,test_size=.1,stratify=y[fi],random_state=20261027)
                run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text())
                assert rec['status']=='complete' and not rec['outer_target_used_in_selection']
                assert rec['runner_sha256']==plan['source_sha256']['research_r10_worker.py']
                assert rec['plan_sha256']==sha(ROOT/'research_r10_plan.json')
                assert rec['inner_split_sha256']==sha(run/'inner_split.npz')
                split=np.load(run/'inner_split.npz')
                for key,expected in [('train',it),('valid',iv),('outer_train',fi),('outer_valid',vi)]:assert np.array_equal(split[key],expected)
                assert np.array_equal(np.sort(np.r_[it,iv]),fi) and not np.intersect1d(np.r_[it,iv],vi).size
                scores=np.array([row['one_minus_auc'] for row in rec['inner_validation']]);selected=rec['selected_epoch']
                assert len(scores)==9 and selected==int(np.flatnonzero(scores==scores.min())[-1])+1
                assert len(rec['outer_epochs'])==selected and rec['peak_gpu_process_bytes']<=CEILING
                for row in rec['outer_epochs']:
                    assert row['training_parameter_devices']==['cuda:0'] and row['backing_data_device']=='cpu'
                    assert row['peak_allocated_bytes']<=CAP
                for role,idx,validation_idx,filename,transformer_name,prediction_file in [
                    ('inner',it,iv,'selection_model.pkl','inner_transformer.joblib','inner_validation.npz'),
                    ('outer',fi,vi,'model.pkl','transformer.joblib','validation.npz')]:
                    expected_hash=rec['selection_model_sha256'] if role=='inner' else rec['model_sha256']
                    expected_tr=rec['inner_transformer_sha256'] if role=='inner' else rec['transformer_sha256']
                    assert sha(run/filename)==expected_hash and sha(run/transformer_name)==expected_tr
                    tr=joblib.load(run/transformer_name);maps(tr,raw,y,idx)
                    saved=np.load(run/prediction_file)
                    assert np.array_equal(saved['id'],df.id.iloc[validation_idx]) and np.array_equal(saved['target'],y[validation_idx])
                    with open(run/filename,'rb') as handle:model=cloudpickle.load(handle)
                    cfg=model.get_config()
                    for key,value in plan['config'].items():
                        if key!='device':assert cfg[key]==value,(role,key)
                    module=model.cv_alg_interface_.model
                    assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==8 and module.progress.max_epochs==9
                    assert module.creator.is_cv==(role=='inner')
                    if role=='inner':
                        assert 'stop_epoch' not in module.config
                        assert model.cv_alg_interface_.fit_params[0]['stop_epoch']['1-auc_ovr']==selected
                    else:assert module.config['stop_epoch']==selected
                    V=tr.transform(raw.iloc[validation_idx[:2048]].reset_index(drop=True));model.to('cuda:0')
                    with threadpool_limits(6):replay=model.predict_proba(V)[:,1]
                    error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<1e-7
                    models.append(dict(stage=stage,fold=k,role=role,selected_epoch=selected,model_sha256=expected_hash,prediction_replay_max_error=error))
                    if role=='outer':nn[vi]=saved['pred']
                    del model,module,tr,V;gc.collect();torch.cuda.empty_cache()
                assert sha(run/'validation.npz')==rec['prediction_sha256']
            assert np.max(np.abs(nn-z['pred']))<1e-12
            if stage=='fivefold':
                from research_nested_stack import outer_predictions
                core=outer_predictions('oof',np.arange(len(df)),df,fold)
                deep=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert np.array_equal(deep['id'],df.id)
                core=np.column_stack([core,deep['pred']])
            else:
                base=np.load(ROOT/'research_artifacts/r4_baseline_replication3/oof.npz')
                assert np.array_equal(base['id'],df.id) and np.array_equal(base['fold'],fold);core=base['components'][:,:4]
            direct=(core@np.array([45,9,18,24])+32*old['pred'])/128
            control=(core@np.array([30,6,12,16])+64*old['pred'])/128
            candidate=(core@np.array([30,6,12,16])+64*nn)/128
            assert np.max(np.abs(direct-old['candidate']))<1e-12 and np.max(np.abs(control-z['reference_half']))<1e-12
            assert np.max(np.abs(candidate-z['candidate']))<1e-12
            checks[stage]=dict(vs_v7=evidence(y,direct,candidate,fold),vs_v8=evidence(y,control,candidate,fold))
            assert all(value['eligible_for_new_partition_replication'] for value in checks[stage].values())
        result=dict(status='passed',audited_utc=now(),data_sha256=HASHES,verified_model_records=len(models),
            verified_native_neural_members=8*len(models),model_records=models,validation=checks,
            outer_validation_excluded_from_selection_and_training=True,inner_TE_and_counts_fit_innertrain_only=True,
            native_unlabelled_inner_val_covariates_allowed_only_within_outertrain=True,
            selected_epoch_schedule_preserved=True,formula='.5v5full+.5mean(five outer native8 ensembles)',
            plan_sha256=sha(ROOT/'research_r10_plan.json'),gpu_process_peak_bytes=peak[0],actual_top100_verified=False,
            limitation='Validation labels historically reused; this audit verifies implementation and conditional evidence, not unselected generalization or leaderboard rank.')
        dump(ROOT/'research_r10_pre_inference_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('model_records','validation')},indent=2))
    finally:stop.set();watcher.join(timeout=5)

if __name__=='__main__':main()

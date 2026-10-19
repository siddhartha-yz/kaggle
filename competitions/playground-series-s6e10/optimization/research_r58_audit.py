"""Independent eight-model audit for the composite-ranking-loss NN74 route."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc,json,threading,inspect,hashlib
import numpy as np,pandas as pd,joblib,cloudpickle,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures
from research_r23_loader import auxiliary as oldaux,pool
from research_r58_loader import auxiliary

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    ap=json.loads((ROOT/'research_r58_audit_plan.json').read_text());plan=json.loads((ROOT/'research_r58_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r58_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['audit_code_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r58_driver_state.json').read_text())['status']=='qualified_for_audit'
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r58_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65
        source=ROOT/'research_artifacts/r20_aux_officialcb';r=json.loads((source/'result.json').read_text());assert r['status']=='complete' and len(r['models'])==65 and not r['satisfaction_column_loaded'] and not r['satisfaction_used_as_target_or_input']
        for row in r['models']:assert sha(source/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        proof=json.loads((ROOT/'research_r58_loss_test.json').read_text());assert proof['status']=='passed_no_official_data' and proof['gradient_error']<1e-12 and proof['loss_error']<1e-12
        df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');aux=auxiliary(df.id);old=oldaux(df.id);assert np.array_equal(aux.iloc[:,13:].to_numpy(),old.iloc[:,13:].to_numpy())
        records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r58_ranking_lean_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(df)),y)):fold[vi]=k
            assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold) and all(np.array_equal(z[n],base[n]) and np.array_equal(z[n],tree[n]) for n in ['id','target','fold'])
            pred=np.full(len(df),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['config']==plan['config'] and rec['plan_sha256']==sha(ROOT/'research_r58_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r58_worker.py']
                assert rec['model_sha256']==sha(run/'model.pkl') and rec['transformer_sha256']==sha(run/'transformer.joblib') and rec['train_rows']==len(fi) and rec['valid_rows']==len(vi) and not rec['validation_used_for_fit_or_epoch_selection']
                assert rec['observed_batch_devices']==['cuda:0'] and len(rec['epochs'])==3 and rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<=8589934592
                assert all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=778043392 for e in rec['epochs'])
                loss=rec['ranking_loss'];assert loss['calls']>0 and loss['first_logits_shape']==[32,256,2] and loss['CE_coefficient']==.9 and loss['pairwise_coefficient']==.1
                tr=joblib.load(run/'transformer.joblib');assert tr.columns==list(raw) and tr.encoder.cv==5 and tr.encoder.random_state==42 and abs(tr.encoder.target_mean_-y[fi].mean())<1e-15 and tr.aux_columns==list(aux)
                keys=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True),cols) for j,cols in enumerate(COMBOS)})
                with threadpool_limits(6):encoder=TargetEncoder(**tr.encoder.get_params()).fit(keys,y[fi])
                assert encoder.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(a,b) for a,b in zip(encoder.categories_,tr.encoder.categories_)) and all(np.array_equal(a,b) for a,b in zip(encoder.encodings_,tr.encoder.encodings_));del encoder,keys
                for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
                query=vi[:2048];V=tr.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),old.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V,C,check_exact=True);assert V.shape==(len(query),74) and V.select_dtypes(include='number').shape[1]==52 and np.array_equal(V.iloc[:,-27:].to_numpy(),aux.iloc[query].to_numpy('float32'))
                with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
                module=model.cv_alg_interface_.model;assert module.criterion.__module__=='research_r58_ranking_loss' and module.criterion.__name__=='ranking_criterion';assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3 and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu'
                saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi])
                with threadpool_limits(6):replay=model.predict_proba(V)[:,1]
                error=float(np.max(np.abs(replay-saved['pred'][:len(query)])));assert error<2e-6;pred[vi]=saved['pred'];records.append(dict(stage=stage,fold=k,native_members=32,CPU_replay_error=error,model_sha256=rec['model_sha256'],own_full_encoder_reconstructed=True,unchanged74fields_exact=True));del tr,V,control,C,model,module,saved;gc.collect()
            candidate=pool(tree['tree'],pred);formal=pool(tree['tree'],base['pred']);assert np.array_equal(pred,z['pred']) and np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(formal-base['candidate']))<1e-12 and np.array_equal(z['reference_v11'],base['candidate']);comparison=evidence(y,formal,candidate,fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r58_plan.json'),audit_plan_sha256=sha(ROOT/'research_r58_audit_plan.json'),verified_model_records=8,verified_native_neural_members=256,model_records=records,validation=validation,composite_loss_math_test_sha256=sha(ROOT/'research_r58_loss_test.json'),reused_original_auxiliary_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),rechecked_auxiliary_model_hashes=65,positive_class='True=1',CPU_threads=6,peak_RSS_bytes=peak[0],sklearn_fit_transform_source_sha256=hashlib.sha256(inspect.getsource(TargetEncoder.fit_transform).encode()).hexdigest(),actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r58_pre_inference_audit.json',result);print('R56_INDEPENDENT_AUDIT_PASSED',flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

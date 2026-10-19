"""Independent own-fold/strict-OOF-input, pureloss, origin and probability audit."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_r23_loader import auxiliary,pool
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures
from research_r68_pure_loss import pure_ranknet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from threadpoolctl import threadpool_limits
from scipy.special import expit,logit
import json,threading,gc,joblib,cloudpickle,numpy as np,pandas as pd,torch

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1);plan=json.loads((ROOT/'research_r68_plan.json').read_text());auditplan=json.loads((ROOT/'research_r68_audit_plan.json').read_text());assert auditplan['main_plan_sha256']==sha(ROOT/'research_r68_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**auditplan['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r68_driver_state.json').read_text())['status']=='qualified_for_audit'
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r68_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65;source=ROOT/'research_artifacts/r20_aux_officialcb';manifest=json.loads((source/'result.json').read_text());assert len(manifest['models'])==65 and not manifest['satisfaction_column_loaded'] and not manifest['satisfaction_used_as_target_or_input']
        for row in manifest['models']:assert sha(source/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        frame=pd.read_csv(ROOT/'data/train.csv');assert len(frame)==699635 and frame.satisfaction.dtype==bool;raw=frame.drop(columns=['id','satisfaction']);y=frame.satisfaction.to_numpy('int8');aux=auxiliary(frame.id);records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r68_pure_ranknet_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=np.full(len(y),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
            for a in [z,base,tree]:assert np.array_equal(a['id'],frame.id) and np.array_equal(a['target'],y) and np.array_equal(a['fold'],fold)
            pred=np.full(len(y),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['config']==plan['config'] and rec['plan_sha256']==sha(ROOT/'research_r68_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r68_worker.py']
                for n,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('validation.npz','prediction_sha256'),('calibration.json','calibration_sha256'),('calibration_training.npz','calibration_training_sha256')]:assert sha(run/n)==rec[key]
                assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi) and not rec['validation_used_for_fit_or_epoch_selection'] and rec['observed_batch_devices']==['cuda:0'] and len(rec['epochs'])==3 and rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<=8589934592
                assert all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=778043392 for e in rec['epochs']);assert rec['ranking_loss']['CE_coefficient']==0. and rec['ranking_loss']['pairwise_coefficient']==1. and rec['ranking_loss']['first_logits_shape']==[32,256,2]
                tr=joblib.load(run/'transformer.joblib');assert tr.columns==list(raw) and tr.aux_columns==list(aux) and tr.encoder.cv==5 and tr.encoder.random_state==42 and tr.encoder.target_type=='binary'
                cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True),columns) for j,columns in enumerate(COMBOS)})
                # Independent sklearn strict-OOF fit_transform reconstructs training TE, not full-encoder transforms.
                with threadpool_limits(6):encoder=TargetEncoder(**tr.encoder.get_params());oof_te=encoder.fit_transform(cross,y[fi])
                assert encoder.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(a,b) for a,b in zip(encoder.categories_,tr.encoder.categories_)) and all(np.array_equal(a,b) for a,b in zip(encoder.encodings_,tr.encoder.encodings_));del encoder,cross
                for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
                query=vi[:2048];V=tr.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V,C,check_exact=True);assert V.shape==(len(query),74) and V.select_dtypes(include='number').shape[1]==52
                fitquery=fi[:2048];T=tr.transform(raw.iloc[fitquery].reset_index(drop=True),aux.iloc[fitquery].reset_index(drop=True))
                for j in range(3):T[f'cross_te_{j}']=oof_te[:len(fitquery),j].astype('float32')
                del oof_te
                with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
                module=model.cv_alg_interface_.model;assert module.criterion is pure_ranknet and np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and module.progress.max_epochs==3 and not module.creator.is_cv and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu'
                saved=np.load(run/'validation.npz');training=np.load(run/'calibration_training.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],frame.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]) and np.array_equal(training['idx'],fi) and np.array_equal(training['id'],frame.id.iloc[fi]) and np.array_equal(training['target'],y[fi])
                with threadpool_limits(6):vp=model.predict_proba(V)[:,1];tp=model.predict_proba(T)[:,1]
                verr=float(np.max(np.abs(vp-saved['raw_pred'][:len(query)])));terr=float(np.max(np.abs(tp-training['raw_pred'][:len(fitquery)])));assert verr<2e-6 and terr<2e-6
                calibration=json.loads((run/'calibration.json').read_text());assert calibration['rows']==len(fi) and calibration['external_query_chunk']==2048 and calibration['clip_epsilon']==1e-6 and not calibration['per_member_origin_invariance_claim'] and calibration['training_prior']==float(y[fi].mean())
                raw_train=logit(np.clip(training['raw_pred'].astype('float64'),1e-6,1-1e-6));lo,hi=-50.,50.
                for _ in range(80):
                    mid=(lo+hi)/2
                    if expit(raw_train+mid).mean()<y[fi].mean():lo=mid
                    else:hi=mid
                root_error=float(abs((lo+hi)/2-calibration['bias']));assert root_error<1e-10 and abs(expit(raw_train+calibration['bias']).mean()-y[fi].mean())<1e-12
                actual=expit(logit(np.clip(saved['raw_pred'].astype('float64'),1e-6,1-1e-6))+calibration['bias']);assert np.array_equal(actual,saved['pred']);pred[vi]=actual
                records.append(dict(stage=stage,fold=k,model_sha256=rec['model_sha256'],calibration_sha256=rec['calibration_sha256'],native_members=32,own_strict_OOF_training_TE_reconstructed=True,unchanged74queryfields_exact=True,CPU_validation_raw_replay_error=verr,CPU_training_raw_replay_error=terr,independent_bisection_origin_error=root_error));print('AUDITED',stage,k,verr,terr,root_error,flush=True);del tr,V,C,T,control,model,module,saved,training,raw_train;gc.collect()
            old=np.clip(base['candidate'],1e-6,1-1e-6);new=np.clip(pred,1e-6,1-1e-6);numerator=old**.875*new**.125;candidate=numerator/(numerator+(1-old)**.875*(1-new)**.125);assert np.array_equal(pred,z['pred']) and np.max(np.abs(candidate-z['candidate']))<1e-12 and np.array_equal(z['reference_v11'],base['candidate']) and np.max(np.abs(pool(tree['tree'],base['pred'])-base['candidate']))<1e-12;comparison=evidence(y,base['candidate'],z['candidate'],fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        assert len(records)==8;report=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r68_plan.json'),audit_plan_sha256=sha(ROOT/'research_r68_audit_plan.json'),verified_model_records=8,verified_native_neural_members=256,model_records=records,validation=validation,rechecked_auxiliary_model_hashes=65,prior_source_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),positive_class='True=1',CPU_threads=6,peak_RSS_bytes=peak[0],actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r68_pre_inference_audit.json',report)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

"""Independent fixed-six-view audit; never fits/selects a classifier or changes gates."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r23_loader import auxiliary as baseline_aux,pool
from research_r40_worker import watch_cpu
from research_r4_lean_features import COMBOS
from research_features import DistanceFeatures
import json,threading,gc,importlib
import numpy as np,pandas as pd,joblib,cloudpickle,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from threadpoolctl import threadpool_limits
from pytabkit.models.training.metrics import cross_entropy

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    plan=json.loads((ROOT/'research_r64_plan.json').read_text());execution=json.loads((ROOT/'research_r64_execution_plan.json').read_text());ap=json.loads((ROOT/'research_r64_audit_plan.json').read_text())
    assert ap['main_plan_sha256']==sha(ROOT/'research_r64_plan.json') and ap['execution_plan_sha256']==sha(ROOT/'research_r64_execution_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**execution['source_sha256'],**execution['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r64_driver_state.json').read_text())['status']=='qualified_for_audit'
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r64_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        # Rank algebra and all65 original pretext posterior class mappings replayed independently on CPU.
        from research_r47_rank_audit import verify_ranks
        rank=verify_ranks();assert rank['status']=='passed';dump(ROOT/'research_r64_rank_audit.json',rank)
        aux_sources=[]
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65
        old=ROOT/'research_artifacts/r20_aux_officialcb';manifest=json.loads((old/'result.json').read_text());assert not manifest['satisfaction_column_loaded'] and not manifest['satisfaction_used_as_target_or_input']
        for row in manifest['models']:assert sha(old/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        aux_sources.append(dict(models=65,source='official covariates excluding SAT',prior_CPU_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json')))
        ca=json.loads((ROOT/'research_r56_auxiliary_audit.json').read_text());assert ca['status'].startswith('passed')
        cohort=ROOT/'research_artifacts/r46_cohort_auxiliary';cm=json.loads((cohort/'result.json').read_text())
        for row in cm['models']:assert sha(cohort/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        aux_sources.append(dict(models=65,source='official nonrating covariates excluding SAT',prior_CPU_audit_sha256=sha(ROOT/'research_r56_auxiliary_audit.json')))
        for route,directory in [('r60','r60_original_classconditional'),('r62','r62_original_cohortclass')]:
            ga=json.loads((ROOT/f'research_{route}_generator_audit.json').read_text());assert ga['status']=='passed_no_official_SAT_or_quality' and ga['verified_source_models']==13 and not ga['scope']['official_SAT_values_parsed'] and ga['scope']['source_SAT_used_as_conditioning_input']
            source=ROOT/'research_artifacts'/directory;manifest=json.loads((source/'result.json').read_text());assert ga['source_manifest_sha256']==sha(source/'result.json')
            for j,row in enumerate(manifest['models']):assert row['model_sha256']==sha(source/f'rating{j}/model.cbm')
            aux_sources.append(dict(models=13,source='CC0 original source SAT-conditioned rating models',prior_CPU_audit_sha256=sha(ROOT/f'research_{route}_generator_audit.json')))
        frame=pd.read_csv(ROOT/'data/train.csv');raw=frame.drop(columns=['id','satisfaction']);y=frame.satisfaction.to_numpy('int8');assert frame.satisfaction.dtype==bool and len(y)==699635
        baseaux=baseline_aux(frame.id);records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r64_viewbag_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz')
            aggregate=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
            fold=np.full(len(y),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
            for z in (aggregate,base,tree):assert np.array_equal(z['id'],frame.id) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],fold)
            views=[]
            for member in plan['members']:
                name=member['name'];aux=importlib.import_module(member['loader_module']).auxiliary(frame.id)
                directory=f"{member['directory']}_{stage}" if stage=='fivefold' or name=='r23' else f'r64_{name}_{stage}'
                src=ROOT/'research_artifacts'/directory;z=np.load(src/'oof.npz');assert all(np.array_equal(z[n],base[n]) for n in ('id','target','fold'));pred=np.full(len(y),np.nan)
                for k in range(K):
                    fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);run=src/f'fold{k}';rec=json.loads((run/'result.json').read_text())
                    ownplan=ROOT/(f'research_{name}_plan.json' if stage=='fivefold' or name=='r23' else 'research_r64_execution_plan.json');op=json.loads(ownplan.read_text())
                    assert rec['status']=='complete' and rec['config']==plan['config'] and rec['plan_sha256']==sha(ownplan)
                    runner=f'research_{name}_worker.py' if stage=='fivefold' or name=='r23' else 'research_r64_worker.py';assert rec['runner_sha256']==op['source_sha256'][runner]
                    for f,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256')]:assert sha(run/f)==rec[key]
                    assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi) and not rec['validation_used_for_fit_or_epoch_selection']
                    assert rec['observed_batch_devices']==['cuda:0'] and len(rec['epochs'])==3 and rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<=8589934592
                    assert all(e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' and e['peak_allocated_bytes']<=778043392 for e in rec['epochs'])
                    tr=joblib.load(run/'transformer.joblib');assert tr.columns==list(raw) and tr.encoder.cv==5 and tr.encoder.random_state==42 and tr.aux_columns==list(aux)
                    keys=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[fi].reset_index(drop=True),columns) for j,columns in enumerate(COMBOS)})
                    with threadpool_limits(6):encoder=TargetEncoder(**tr.encoder.get_params()).fit(keys,y[fi])
                    assert encoder.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(a,b) for a,b in zip(encoder.categories_,tr.encoder.categories_)) and all(np.array_equal(a,b) for a,b in zip(encoder.encodings_,tr.encoder.encodings_));del encoder,keys
                    for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[fi][c].value_counts().sort_index())
                    query=vi[:2048];V=tr.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));control=joblib.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/fold{k}/transformer.joblib');C=control.transform(raw.iloc[query].reset_index(drop=True),baseaux.iloc[query].reset_index(drop=True));assert V.shape==(len(query),74) and V.select_dtypes(include='number').shape[1]==52
                    if name in ('r56','r61','r63'):
                        pd.testing.assert_frame_equal(V.iloc[:,:47],C.iloc[:,:47],check_exact=True);pd.testing.assert_frame_equal(V.iloc[:,60:],C.iloc[:,60:],check_exact=True);assert np.array_equal(V.iloc[:,-27:].to_numpy(),aux.iloc[query].to_numpy('float32'))
                    elif name=='r47':
                        from research_r47_features import RATINGS
                        unchanged=[c for c in V if c not in RATINGS];pd.testing.assert_frame_equal(V[unchanged],C[unchanged],check_exact=True)
                        for j,c in enumerate(RATINGS):assert np.array_equal(V[c].to_numpy(),aux.iloc[query,27+j].to_numpy('float32'))
                    elif name=='r59':
                        from research_r59_features import DELAYS
                        unchanged=[c for c in V if c not in DELAYS];pd.testing.assert_frame_equal(V[unchanged],C[unchanged],check_exact=True)
                        for c in DELAYS:assert np.array_equal(V[c].to_numpy(),np.log1p(C[c].to_numpy('float64')).astype('float32'))
                    else:pd.testing.assert_frame_equal(V,C,check_exact=True)
                    with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
                    module=model.cv_alg_interface_.model;assert module.criterion is cross_entropy and np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3 and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu'
                    saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],frame.id.iloc[vi]) and np.array_equal(saved['target'],y[vi])
                    with threadpool_limits(6):replay=model.predict_proba(V)[:,1]
                    error=float(np.max(np.abs(replay-saved['pred'][:len(query)])));assert error<2e-6;pred[vi]=saved['pred'];records.append(dict(stage=stage,view=name,fold=k,native_members=32,CPU_replay_error=error,own_full_encoder_reconstructed=True,model_sha256=rec['model_sha256'],transformer_sha256=rec['transformer_sha256']))
                    print('AUDITED',stage,name,k,error,flush=True);del tr,V,control,C,model,module,saved;gc.collect()
                assert np.array_equal(pred,z['pred']);views.append(pred);del aux;gc.collect()
            mean=np.mean(views,axis=0);candidate=pool(tree['tree'],mean);assert np.array_equal(mean,aggregate['pred']) and np.max(np.abs(candidate-aggregate['candidate']))<1e-12 and np.max(np.abs(pool(tree['tree'],base['pred'])-base['candidate']))<1e-12
            comparison=evidence(y,base['candidate'],candidate,fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        assert len(records)==48
        report=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r64_plan.json'),execution_plan_sha256=sha(ROOT/'research_r64_execution_plan.json'),audit_plan_sha256=sha(ROOT/'research_r64_audit_plan.json'),verified_model_records=len(records),verified_native_neural_members=len(records)*32,model_records=records,validation=validation,auxiliary_sources=aux_sources,rank_audit_sha256=sha(ROOT/'research_r64_rank_audit.json'),positive_class='True=1',CPU_threads=6,peak_RSS_bytes=peak[0],source_scope='Original public-source SAT used to condition R61/R63 auxiliary classifiers; official SAT excluded from all auxiliary generators. Legacy training-record auxiliary_target_free_transductive means official SAT excluded and does not assert absence of source SAT.',actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r64_pre_inference_audit.json',report)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

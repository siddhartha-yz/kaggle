"""Independent nested-membership, preprocessing, CPU replay and logit algebra audit."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc,json,hashlib,joblib,numpy as np,pandas as pd,torch
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r32_features import ConditionalTabMFeatures
from research_r23_loader import auxiliary,pool
from research_r32_engine import load,predict
# Engine import sets CUDA visibility but this audit executes CPU-only operations.
os.environ['CUDA_VISIBLE_DEVICES']=''

def own_maps(actual,raw,y,aux):
    expected=ConditionalTabMFeatures();expected.fit_transform(raw.reset_index(drop=True),y,aux.reset_index(drop=True))
    a,b=actual.lean,expected.lean
    assert a.columns==b.columns and a.cats==b.cats and a.numeric==b.numeric and a.aux_columns==b.aux_columns and a.output_columns==b.output_columns
    for k in b.counts:pd.testing.assert_series_equal(a.counts[k],b.counts[k])
    assert a.encoder.cv==5 and a.encoder.random_state==42 and a.encoder.target_type=='binary'
    assert a.encoder.target_mean_==b.encoder.target_mean_
    for x,z in zip(a.encoder.categories_,b.encoder.categories_):assert np.array_equal(x,z)
    for x,z in zip(a.encoder.encodings_,b.encoder.encodings_):assert np.array_equal(x,z)
    a,b=actual.tabm,expected.tabm
    assert a.columns==b.columns and a.num==b.num and a.cat==b.cat and a.vocabs==b.vocabs and a.cardinalities==b.cardinalities
    pd.testing.assert_series_equal(a.medians,b.medians);assert np.array_equal(a.mask,b.mask)
    assert np.array_equal(a.qt.quantiles_,b.qt.quantiles_) and np.array_equal(a.qt.references_,b.qt.references_)
    del expected;gc.collect()

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    final=json.loads((ROOT/'research_r32_final_inference_plan.json').read_text());plan=json.loads((ROOT/'research_r32_plan.json').read_text());assert final['main_plan_sha256']==sha(ROOT/'research_r32_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**final['code_sha256'],**final['inherited_artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    prior=json.loads((ROOT/'research_r23_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['verified_model_records']==8
    prioraux=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prioraux['status']=='passed' and prioraux['source_rating_models_verified']==65
    df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool
    raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');aux=auxiliary(df.id);records=[];checks={}
    with threadpool_limits(6):
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r32_tabm74_{stage}';decision=json.loads((out/'result.json').read_text());assert decision['eligible'];z=np.load(out/'oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');base=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz')
            fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(raw,y)):fold[vi]=k
            assert all(np.array_equal(v['id'],df.id) and np.array_equal(v['target'],y) and np.array_equal(v['fold'],fold) for v in [z,nn,base]);pred=np.full(len(df),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.1,stratify=y[fi],random_state=20261018);run=out/f'fold{k}';rec=json.loads((run/'result.json').read_text());sel=json.loads((run/'selection.json').read_text());fitrec=json.loads((run/'model.json').read_text());spl=np.load(run/'inner_split.npz')
                assert np.array_equal(spl['fit_global'],it) and np.array_equal(spl['valid_global'],im) and not np.intersect1d(np.r_[it,im],vi).size
                assert rec['status']=='complete' and rec['feature_width']==74 and not rec['outer_validation_used_for_fit_or_selection'] and rec['runner_sha256']==plan['source_sha256']['research_r32_worker.py'] and rec['plan_sha256']==sha(ROOT/'research_r32_plan.json')
                assert rec['train_rows']==len(fi) and rec['valid_rows']==len(vi) and rec['train_index_sha256']==hashlib.sha256(fi.tobytes()).hexdigest()
                best=rec['selected_epochs'];assert 1<=best<=20 and best==sel['best_epoch']==int(np.argmax([r['inner_validation_auc'] for r in sel['curve']]))+1
                assert sel['config']==plan['config'] and len(sel['curve'])<=20 and fitrec['config']==dict(plan['config'],epochs=best) and len(fitrec['curve'])==best
                for rr in [sel,fitrec]:assert rr['parameters_device']=='cuda:0' and rr['backing_device']=='cpu' and rr['peak_allocated_bytes']<=778043392
                for key,name in [('inner_split_sha256','inner_split.npz'),('selection_transformer_sha256','selection_transformer.joblib'),('selection_model_sha256','selection.pt'),('selection_sha256','selection.json'),('model_sha256','model.pt'),('transformer_sha256','transformer.joblib'),('prediction_sha256','validation.npz')]:assert rec[key]==sha(run/name)
                assert rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<=8589934592
                inner=joblib.load(run/'selection_transformer.joblib');own_maps(inner,raw.iloc[it],y[it],aux.iloc[it]);IM=inner.transform(raw.iloc[im].reset_index(drop=True),aux.iloc[im].reset_index(drop=True));sm=load(run/'selection.pt',device='cpu');ip=predict(sm,IM,device='cpu');inner_error=abs(float(roc_auc_score(y[im],ip))-sel['curve'][-1]['inner_validation_auc']);assert inner_error<2e-6
                del inner,IM,sm,ip;gc.collect()
                tr=joblib.load(run/'transformer.joblib');own_maps(tr,raw.iloc[fi],y[fi],aux.iloc[fi]);V=tr.transform(raw.iloc[vi].reset_index(drop=True),aux.iloc[vi].reset_index(drop=True));saved=np.load(run/'validation.npz')
                assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]);m=load(run/'model.pt',device='cpu');q=(V[0][:2048],V[1][:2048]);replay=predict(m,q,device='cpu');error=float(np.max(np.abs(replay-saved['pred'][:2048])));assert error<2e-5
                ck=torch.load(run/'model.pt',map_location='cpu',weights_only=True);args=ck['args'];assert args['k']==16 and args['n_classes']==2 and args['cat_cardinalities']==tr.cardinalities and args['n_num_features']==V[0].shape[1] and args['arch_type']=='tabm'
                pred[vi]=saved['pred'];records.append(dict(stage=stage,fold=k,selected_epochs=best,model_sha256=rec['model_sha256'],CPU_replay_error=error,inner_last_auc_replay_error=inner_error,native_members=16));del tr,V,m,replay,ck;gc.collect()
            assert np.max(np.abs(pred-z['pred']))<1e-12
            # Independent algebra: current tree was audited in R21 and unchanged here.
            formal=pool(base['tree'],nn['pred']);assert np.max(np.abs(formal-nn['candidate']))<1e-12
            candidate=expit(.5*logit(np.clip(base['tree'],1e-6,1-1e-6))+.375*logit(np.clip(nn['pred'],1e-6,1-1e-6))+.125*logit(np.clip(pred,1e-6,1-1e-6)))
            assert np.max(np.abs(candidate-z['candidate']))<1e-12 and np.max(np.abs(formal-z['reference_v11']))<1e-12
            checks[stage]=evidence(y,formal,candidate,fold);assert checks[stage]['eligible_for_new_partition_replication']
    result=dict(status='passed',audited_utc=now(),plan_sha256=sha(ROOT/'research_r32_plan.json'),verified_new_outer_models=len(records),verified_nested_selection_models=len(records),verified_native_members=16*len(records),model_records=records,validation=checks,all_inner_outer_maps_refitted_and_verified=True,CPU_predictions_replayed=True,independent_logit_algebra_verified=True,source_auxiliary_models_reused=65,positive_class='True=1',final_test_method='Uniformmean5savedoutermodels,80TabMmembers;tree/NN74testunchanged',actual_top500_verified=False,limitation='OfficialSATvalidationlabelreuse;conditionalboundsomitadaptivehistory.')
    dump(ROOT/'research_r32_pre_inference_audit.json',result);print('AUDIT_PASS',len(records),flush=True)
if __name__=='__main__':main()

"""Independent nested splits, all inner expert replays, pairwise optimum and outer algebra."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc,json,threading,hashlib,joblib,cloudpickle,numpy as np,pandas as pd,torch,lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import TargetEncoder
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_external_ablation import external
from research_r23_loader import auxiliary
from research_features import DistanceFeatures
from research_r4_lean_features import COMBOS
from research_r68_pure_loss import pure_ranknet

def neural(run,it,im,raw,y,aux,plan,ranking):
    rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['config']==plan['config'] and rec['training_index_sha256']==hashlib.sha256(it.tobytes()).hexdigest() and rec['train_rows']==len(it) and rec['valid_rows']==len(im)
    assert not rec['validation_used_for_fit_or_epoch_selection'] and len(rec['epochs'])==3 and rec['observed_batch_devices']==['cuda:0'] and rec['peak_gpu_process_bytes']<=1073741824 and rec['peak_rss_bytes']<=8589934592
    assert all(e['peak_allocated_bytes']<=778043392 and e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' for e in rec['epochs'])
    for file,key in [('model.pkl','model_sha256'),('transformer.joblib','transformer_sha256'),('validation.npz','prediction_sha256')]:assert sha(run/file)==rec[key]
    tr=joblib.load(run/'transformer.joblib');assert tr.columns==list(raw) and tr.aux_columns==list(aux) and tr.encoder.cv==5 and tr.encoder.random_state==42
    cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[it].reset_index(drop=True),c) for j,c in enumerate(COMBOS)})
    encoder=TargetEncoder(**tr.encoder.get_params());oof=encoder.fit_transform(cross,y[it]);assert encoder.target_mean_==tr.encoder.target_mean_ and all(np.array_equal(a,b) for a,b in zip(encoder.categories_,tr.encoder.categories_)) and all(np.array_equal(a,b) for a,b in zip(encoder.encodings_,tr.encoder.encodings_));del encoder,cross
    for c,counts in tr.counts.items():assert counts.sort_index().equals(raw.iloc[it][c].value_counts().sort_index())
    query=im[:2048];N=tr.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));assert N.shape==(len(query),74) and N.select_dtypes(include='number').shape[1]==52
    saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],im) and np.array_equal(saved['target'],y[im])
    with open(run/'model.pkl','rb') as f:model=cloudpickle.load(f)
    module=model.cv_alg_interface_.model;assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3 and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu'
    replay=model.predict_proba(N)[:,1];error=float(np.max(np.abs(replay-saved['raw_pred' if ranking else 'pred'][:len(query)])));assert error<2e-6
    result=dict(CPU_meta_replay_error=error,model_sha256=rec['model_sha256'],strict_OOF_training_encoder_reconstructed=True)
    if ranking:
        assert module.criterion is pure_ranknet and rec['ranking_loss']['CE_coefficient']==0 and rec['ranking_loss']['pairwise_coefficient']==1
        for file,key in [('calibration.json','calibration_sha256'),('calibration_training.npz','calibration_training_sha256')]:assert sha(run/file)==rec[key]
        train=np.load(run/'calibration_training.npz');assert np.array_equal(train['idx'],it) and np.array_equal(train['target'],y[it]);T=tr.transform(raw.iloc[it[:2048]].reset_index(drop=True),aux.iloc[it[:2048]].reset_index(drop=True))
        for j in range(3):T[f'cross_te_{j}']=oof[:len(T),j].astype('float32')
        terr=float(np.max(np.abs(model.predict_proba(T)[:,1]-train['raw_pred'][:len(T)])));assert terr<2e-6
        cal=json.loads((run/'calibration.json').read_text());assert cal['rows']==len(it) and cal['training_prior']==float(y[it].mean()) and cal['external_query_chunk']==2048 and cal['clip_epsilon']==1e-6 and not cal['per_member_origin_invariance_claim'];s=logit(np.clip(train['raw_pred'].astype('float64'),1e-6,1-1e-6));lo,hi=-50.,50.
        for _ in range(80):
            mid=(lo+hi)/2
            if expit(s+mid).mean()<y[it].mean():lo=mid
            else:hi=mid
        berr=float(abs((lo+hi)/2-cal['bias']));assert berr<1e-10 and np.array_equal(saved['pred'],expit(logit(np.clip(saved['raw_pred'].astype('float64'),1e-6,1-1e-6))+cal['bias']));result.update(CPU_training_replay_error=terr,independent_intercept_error=berr)
    del model,module,tr,N,oof;gc.collect();return result

def tree_replays(path,it,im,raw,ext,teacher,aux,components):
    rec=json.loads((path/'result.json').read_text());assert rec['status']=='complete' and rec['train_rows']==len(it) and rec['meta_rows']==len(im) and rec['outer_validation_excluded'];q=im[:2048]
    fresh=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[it].reset_index(drop=True));fresh.te_maps=[];V=fresh.transform_te(raw.iloc[q].reset_index(drop=True))
    for c in ext:V[c]=ext[c].to_numpy()[q]
    saved=joblib.load(path/'old57_transformer.joblib');assert saved.n==len(it) and saved.te_maps==[] and rec['old57_transformer_sha256']==sha(path/'old57_transformer.joblib');S=saved.transform_te(raw.iloc[q].reset_index(drop=True))
    for c in ext:S[c]=ext[c].to_numpy()[q]
    pd.testing.assert_frame_equal(V,S,check_exact=True);V84=V.copy()
    for c in aux:V84[c]=aux[c].to_numpy('float32')[q]
    tr84=joblib.load(path/'lgb84_transformer.joblib');assert tr84.base.n==len(it) and tr84.base.te_maps==[] and rec['models']['lgb84']['transformer_sha256']==sha(path/'lgb84_transformer.joblib');S84=tr84.transform(raw.iloc[q].reset_index(drop=True),ext.iloc[q].reset_index(drop=True),aux.iloc[q].reset_index(drop=True));pd.testing.assert_frame_equal(V84,S84,check_exact=True);errors={}
    for j,name in enumerate(['lgb84','cb8','residual','cb10']):
        row=rec['models'][name];file=ROOT/row['model'];assert sha(file)==row['model_sha256']
        if name in ['cb8','cb10']:
            model=CatBoostClassifier(thread_count=6);model.load_model(str(file));assert np.array_equal(model.classes_,[0,1]) and model.tree_count_==(1308 if name=='cb8' else 813);Q=V.copy()
            for c in Q.select_dtypes('category'):Q[c]=Q[c].astype(str)
            p=model.predict_proba(Q,thread_count=6,task_type='CPU')[:,1];del Q
        else:
            model=lgb.Booster(model_file=str(file));assert model.num_trees()==(639 if name=='lgb84' else 581) and model.num_feature()==(84 if name=='lgb84' else 57);p=model.predict(V84 if name=='lgb84' else V,raw_score=name=='residual',num_threads=6)
            if name=='residual':p=expit(p+.5*logit(np.clip(teacher[q],1e-6,1-1e-6)))
        error=float(np.max(np.abs(p-components[:len(q),j])));assert error<1e-12;errors[name]=error;del model
    return errors

def scalar_optimum(tree,nn,rank,y,gate):
    t,n,r=[logit(np.clip(p.astype('float64'),1e-6,1-1e-6)) for p in (tree,nn,rank)];m=.875*.5*(t+n)+.125*r;x=.875*(n-t);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(20261069);i=np.r_[np.repeat(pos,32),pos[rng.integers(len(pos),size=len(neg)*32)]];j=np.r_[neg[rng.integers(len(neg),size=len(pos)*32)],np.repeat(neg,32)];a=m[i]-m[j];d=x[i]-x[j];assert gate.optimization['weighted_pairs']==32*len(y) and gate.ridge==100 and gate.k==32 and gate.seed==20261069 and -.5<=gate.theta<=.5
    # Deliberately keep duplicate edges: mathematically identical to multiplicity weights.
    def grad(v):return float(np.sum(-expit(-(a+v*d))*d)/32+100*v)
    lo,hi=-.5,.5
    if grad(lo)>=0:opt=lo
    elif grad(hi)<=0:opt=hi
    else:
        for _ in range(60):
            mid=(lo+hi)/2
            if grad(mid)<0:lo=mid
            else:hi=mid
        opt=(lo+hi)/2
    err=float(abs(opt-gate.theta));assert err<1e-10;return err

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1);plan=json.loads((ROOT/'research_r69_plan.json').read_text());ap=json.loads((ROOT/'research_r69_audit_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r69_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r69_driver_state.json').read_text())['status']=='qualified_for_audit';stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r69_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65;source=ROOT/'research_artifacts/r20_aux_officialcb';manifest=json.loads((source/'result.json').read_text());assert len(manifest['models'])==65 and not manifest['satisfaction_column_loaded'] and not manifest['satisfaction_used_as_target_or_input']
        for row in manifest['models']:assert sha(source/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();aux=auxiliary(df.id);records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r69_rank_gate_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');rank=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz');fold=np.full(len(y),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):fold[vi]=k
            for a in [z,base,tree,rank]:assert np.array_equal(a['id'],df.id) and np.array_equal(a['target'],y) and np.array_equal(a['fold'],fold)
            pred=np.full(len(y),np.nan)
            for k in range(K):
                fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);run=out/f'fold{k}';idx=np.load(run/'inner_indices.npz');assert all(np.array_equal(idx[n],v) for n,v in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)]) and not np.intersect1d(np.r_[it,im],vi).size and not np.intersect1d(it,im).size
                route='r57' if stage=='fivefold' else 'r69';directory='r57_nested_gate' if stage=='fivefold' else 'r69_rank_gate';tp=ROOT/f'research_artifacts/{directory}_{stage}/fold{k}/trees';npth=ROOT/f'research_artifacts/{route}_inner_lean_{stage}/fold{k}';rp=ROOT/f'research_artifacts/r69_inner_ranknet_{stage}/fold{k}';rec=json.loads((run/'result.json').read_text());assert rec['status']=='complete' and rec['plan_sha256']==sha(ROOT/'research_r69_plan.json') and rec['runner_sha256']==plan['source_sha256']['research_r69_gate_worker.py'] and rec['inner_indices_sha256']==sha(run/'inner_indices.npz')
                for path,key in [(tp/'result.json','inner_tree_record_sha256'),(npth/'result.json','inner_nn_record_sha256'),(rp/'result.json','inner_ranknet_record_sha256'),(run/'gate.joblib','gate_sha256'),(run/'validation.npz','validation_sha256')]:assert sha(path)==rec[key]
                tr=json.loads((tp/'result.json').read_text());nr=json.loads((npth/'result.json').read_text());rr=json.loads((rp/'result.json').read_text());routeplan=json.loads((ROOT/f'research_{route}_plan.json').read_text());assert tr['plan_sha256']==nr['plan_sha256']==sha(ROOT/f'research_{route}_plan.json') and rr['plan_sha256']==sha(ROOT/'research_r69_plan.json');assert tr['runner_sha256']==routeplan['source_sha256'][f'research_{route}_tree_worker.py'] and nr['runner_sha256']==routeplan['source_sha256'][f'research_{route}_nn_worker.py'] and rr['runner_sha256']==plan['source_sha256']['research_r69_ranknet_worker.py']
                trees=np.load(tp/'meta_predictions.npz');nn=np.load(npth/'validation.npz');rn=np.load(rp/'validation.npz');assert tr['prediction_sha256']==sha(tp/'meta_predictions.npz')
                for a in [trees,nn,rn]:assert np.array_equal(a['idx'],im) and np.array_equal(a['id'],df.id.iloc[im]) and np.array_equal(a['target'],y[im])
                ct=np.load(rp/'calibration_training.npz');assert np.array_equal(ct['id'],df.id.iloc[it]);del ct
                with threadpool_limits(6):te=tree_replays(tp,it,im,raw,ext,teacher,aux,trees['components']);ne=neural(npth,it,im,raw,y,aux,plan,False);re=neural(rp,it,im,raw,y,aux,plan,True)
                gate=joblib.load(run/'gate.joblib');err=scalar_optimum(trees['components']@np.array(plan['tree_weights']),nn['pred'],rn['pred'],y[im],gate)
                t,n,r=[logit(np.clip(p.astype('float64'),1e-6,1-1e-6)) for p in (tree['tree'][vi],base['pred'][vi],rank['pred'][vi])];actual=expit(.875*((.5-gate.theta)*t+(.5+gate.theta)*n)+.125*r);saved=np.load(run/'validation.npz');assert np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]) and np.array_equal(saved['reference_v12'],rank['candidate'][vi]);assert np.max(np.abs(actual-saved['pred']))<1e-12;pred[vi]=saved['pred'];records.append(dict(stage=stage,fold=k,gate_sha256=rec['gate_sha256'],inner_tree_CPU_replay_errors=te,inner_NNCE=ne,inner_RankNet=re,independent_pair_gate_optimum_error=err,proper_nested_fits=True));print('AUDITED',stage,k,err,flush=True);del trees,nn,rn,gate,saved;gc.collect()
            assert np.array_equal(pred,z['candidate']) and np.array_equal(z['reference_v12'],rank['candidate']);comparison=evidence(y,rank['candidate'],pred,fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        report=dict(status='passed',audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r69_plan.json'),audit_plan_sha256=sha(ROOT/'research_r69_audit_plan.json'),verified_inner_tree_models=32,verified_inner_NNCE_models=8,verified_inner_RankNet_models=8,verified_scalar_gates=8,model_records=records,validation=validation,rechecked_auxiliary_model_hashes=65,CPU_threads=6,peak_RSS_bytes=peak[0],actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r69_pre_inference_audit.json',report)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

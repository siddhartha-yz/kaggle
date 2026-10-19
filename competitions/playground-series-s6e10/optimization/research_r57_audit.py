"""Independent proper nesting, inner expert replays, convex gate optimum and outer algebra."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,threading,hashlib,joblib,cloudpickle,numpy as np,pandas as pd,torch,lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import TargetEncoder
from scipy.special import expit,logit
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha,evidence
from research_r40_worker import watch_cpu
from research_external_ablation import external
from research_features import DistanceFeatures
from research_r4_lean_features import COMBOS
from research_r23_loader import auxiliary,pool
from research_r57_gate import keys,logits

def main():
    torch.set_num_threads(6);torch.set_num_interop_threads(1);ap=json.loads((ROOT/'research_r57_audit_plan.json').read_text());plan=json.loads((ROOT/'research_r57_plan.json').read_text());assert ap['main_plan_sha256']==sha(ROOT/'research_r57_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['audit_code_sha256']}.items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'research_r57_driver_state.json').read_text())['status']=='qualified_for_audit'
    stop=threading.Event();peak=[0];monitor=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r57_audit_resource_failure.json'),daemon=True);monitor.start()
    try:
        prior=json.loads((ROOT/'research_r21_pre_inference_audit.json').read_text());assert prior['status']=='passed' and prior['source_rating_models_verified']==65
        src=ROOT/'research_artifacts/r20_aux_officialcb';r=json.loads((src/'result.json').read_text());assert not r['satisfaction_column_loaded'] and not r['satisfaction_used_as_target_or_input'] and len(r['models'])==65
        for row in r['models']:assert sha(src/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}"/'model.cbm')==row['model_sha256']
        df=pd.read_csv(ROOT/'data/train.csv');assert len(df)==699635 and df.satisfaction.dtype==bool;raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.to_numpy('int8');ext,_=external('both',df.id);teacher=ext.orig_teacher_lgb.to_numpy();aux=auxiliary(df.id);records=[];validation={}
        for stage,K,seed in [('fivefold',5,20261018),('threefold',3,20261022)]:
            out=ROOT/f'research_artifacts/r57_nested_gate_{stage}';result=json.loads((out/'result.json').read_text());assert result['eligible'] and result['oof_sha256']==sha(out/'oof.npz');z=np.load(out/'oof.npz');base=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');outer=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');fold=np.full(len(df),-1,'int8')
            for k,(_,vi) in enumerate(StratifiedKFold(K,shuffle=True,random_state=seed).split(np.zeros(len(df)),y)):fold[vi]=k
            assert np.array_equal(z['id'],df.id) and np.array_equal(z['target'],y) and all(np.array_equal(a['fold'],fold) and np.array_equal(a['id'],df.id) and np.array_equal(a['target'],y) for a in [z,base,outer]);pred=np.full(len(df),np.nan)
            for k in range(K):
                run=out/f'fold{k}';fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);idx=np.load(run/'inner_indices.npz');assert all(np.array_equal(idx[n],a) for n,a in [('outer_train',fi),('outer_valid',vi),('inner_train',it),('inner_meta',im)]) and not np.intersect1d(np.r_[it,im],vi).size and not np.intersect1d(it,im).size
                rec=json.loads((run/'result.json').read_text());treepath=run/'trees';trrec=json.loads((treepath/'result.json').read_text());nrpath=ROOT/f'research_artifacts/r57_inner_lean_{stage}/fold{k}';nr=json.loads((nrpath/'result.json').read_text());assert all(a['status']=='complete' and a['plan_sha256']==sha(ROOT/'research_r57_plan.json') for a in [rec,trrec,nr]);assert rec['inner_indices_sha256']==trrec['inner_indices_sha256']==sha(run/'inner_indices.npz') and rec['inner_tree_record_sha256']==sha(treepath/'result.json') and rec['inner_nn_record_sha256']==sha(nrpath/'result.json')
                assert trrec['runner_sha256']==plan['source_sha256']['research_r57_tree_worker.py'] and nr['runner_sha256']==plan['source_sha256']['research_r57_nn_worker.py'] and rec['runner_sha256']==plan['source_sha256']['research_r57_gate_worker.py'];assert trrec['prediction_sha256']==sha(treepath/'meta_predictions.npz') and nr['prediction_sha256']==rec['inner_nn_prediction_sha256']==sha(nrpath/'validation.npz')
                trees=np.load(treepath/'meta_predictions.npz');nn=np.load(nrpath/'validation.npz');assert all(np.array_equal(a['idx'],im) and np.array_equal(a['id'],df.id.iloc[im]) and np.array_equal(a['target'],y[im]) for a in [trees,nn]);assert nr['training_index_sha256']==hashlib.sha256(it.tobytes()).hexdigest() and not nr['validation_used_for_fit_or_epoch_selection'] and nr['config']==plan['config'] and len(nr['epochs'])==3 and nr['observed_batch_devices']==['cuda:0'] and nr['peak_gpu_process_bytes']<=1073741824 and nr['peak_rss_bytes']<=8589934592 and all(e['peak_allocated_bytes']<=778043392 and e['training_parameter_devices']==['cuda:0'] and e['backing_data_device']=='cpu' for e in nr['epochs'])
                query=im[:2048]
                with threadpool_limits(6):
                    fresh=DistanceFeatures('distance_stats_numeric').fit(raw.iloc[it].reset_index(drop=True));fresh.te_maps=[];V=fresh.transform_te(raw.iloc[query].reset_index(drop=True))
                    for c in ext:V[c]=ext[c].to_numpy()[query]
                    saved57=joblib.load(treepath/'old57_transformer.joblib');assert saved57.n==len(it) and saved57.te_maps==[] and trrec['old57_transformer_sha256']==sha(treepath/'old57_transformer.joblib');S=saved57.transform_te(raw.iloc[query].reset_index(drop=True))
                    for c in ext:S[c]=ext[c].to_numpy()[query]
                    pd.testing.assert_frame_equal(V,S,check_exact=True);V84=V.copy()
                    for c in aux:V84[c]=aux[c].to_numpy('float32')[query]
                    tr84=joblib.load(treepath/'lgb84_transformer.joblib');assert tr84.base.n==len(it) and tr84.base.te_maps==[] and trrec['models']['lgb84']['transformer_sha256']==sha(treepath/'lgb84_transformer.joblib');S84=tr84.transform(raw.iloc[query].reset_index(drop=True),ext.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));pd.testing.assert_frame_equal(V84,S84,check_exact=True);tree_errors={}
                    for j,name in enumerate(['lgb84','cb8','residual','cb10']):
                        row=trrec['models'][name];file=ROOT/row['model'];assert sha(file)==row['model_sha256']
                        if name in ['cb8','cb10']:
                            m=CatBoostClassifier(thread_count=6);m.load_model(str(file));assert np.array_equal(m.classes_,[0,1]) and m.tree_count_==(1308 if name=='cb8' else 813);CV=V.copy()
                            for c in CV.select_dtypes('category'):CV[c]=CV[c].astype(str)
                            replay=m.predict_proba(CV,thread_count=6,task_type='CPU')[:,1];del CV
                        else:
                            m=lgb.Booster(model_file=str(file));assert m.num_trees()==(639 if name=='lgb84' else 581) and m.num_feature()==(84 if name=='lgb84' else 57);replay=m.predict(V84 if name=='lgb84' else V,raw_score=name=='residual',num_threads=6)
                            if name=='residual':replay=expit(replay+.5*logit(np.clip(teacher[query],1e-6,1-1e-6)))
                        err=float(np.max(np.abs(replay-trees['components'][:len(query),j])));assert err<1e-12;tree_errors[name]=err;del m
                    assert nr['model_sha256']==sha(nrpath/'model.pkl') and nr['transformer_sha256']==sha(nrpath/'transformer.joblib');nt=joblib.load(nrpath/'transformer.joblib');assert nt.columns==list(raw) and nt.aux_columns==list(aux) and nt.encoder.cv==5 and nt.encoder.random_state==42 and abs(nt.encoder.target_mean_-y[it].mean())<1e-15
                    cross=pd.DataFrame({f'combo{j}':DistanceFeatures._key(raw.iloc[it].reset_index(drop=True),cols) for j,cols in enumerate(COMBOS)});enc=TargetEncoder(**nt.encoder.get_params()).fit(cross,y[it]);assert all(np.array_equal(a,b) for a,b in zip(enc.encodings_,nt.encoder.encodings_)) and all(np.array_equal(a,b) for a,b in zip(enc.categories_,nt.encoder.categories_));del enc,cross
                    for c,counts in nt.counts.items():assert counts.sort_index().equals(raw.iloc[it][c].value_counts().sort_index())
                    N=nt.transform(raw.iloc[query].reset_index(drop=True),aux.iloc[query].reset_index(drop=True));assert N.shape==(len(query),74) and N.select_dtypes(include='number').shape[1]==52 and np.array_equal(N.iloc[:,-27:].to_numpy(),aux.iloc[query].to_numpy('float32'))
                    with open(nrpath/'model.pkl','rb') as f:model=cloudpickle.load(f)
                    module=model.cv_alg_interface_.model;assert np.array_equal(model.classes_,[0,1]) and module.config['n_ens']==32 and not module.creator.is_cv and module.progress.max_epochs==3 and all(p.device.type=='cpu' for p in module.parameters());model.to('cpu');module.creator.device_info='cpu';nrerr=float(np.max(np.abs(model.predict_proba(N)[:,1]-nn['pred'][:len(query)])));assert nrerr<2e-6
                assert rec['gate_sha256']==sha(run/'gate.joblib');gate=joblib.load(run/'gate.joblib');assert gate.ridge==100. and np.max(np.abs(gate.theta))<=.5 and gate.vocab.tolist()==rec['cohort_vocabulary'] and gate.theta.tolist()==rec['delta_from_equal_weight'];inner_tree=trees['components']@np.array([.46875,.09375,.1875,.25]);a,d=logits(inner_tree,nn['pred']);key=keys(raw.iloc[im]);opt=[]
                for group in gate.vocab:
                    mask=key==group
                    def derivative(theta):return float(np.sum((expit(a[mask]+theta*d[mask])-y[im][mask])*d[mask])+100.*theta)
                    opt.append(-.5 if derivative(-.5)>=0 else .5 if derivative(.5)<=0 else brentq(derivative,-.5,.5,xtol=1e-12))
                optimum_error=float(np.max(np.abs(np.array(opt)-gate.theta)));assert optimum_error<1e-5
                saved=np.load(run/'validation.npz');assert rec['validation_sha256']==sha(run/'validation.npz') and np.array_equal(saved['idx'],vi) and np.array_equal(saved['id'],df.id.iloc[vi]) and np.array_equal(saved['target'],y[vi]);actual=gate.predict(raw.iloc[vi],outer['tree'][vi],base['pred'][vi]);assert np.array_equal(actual,saved['pred']);pred[vi]=actual;records.append(dict(stage=stage,fold=k,inner_tree_CPU_replay_errors=tree_errors,inner_NN_CPU_replay_error=nrerr,independent_scalar_gate_optimum_error=optimum_error,outer_prediction_exact=True,gate_sha256=rec['gate_sha256'],proper_inner_only_expert_and_gate_fits=True));del fresh,V,V84,S,S84,saved57,tr84,nt,N,model,module,gate,trees,nn;gc.collect()
            assert np.array_equal(pred,z['candidate']) and np.array_equal(z['reference_v11'],base['candidate']) and np.max(np.abs(pool(outer['tree'],base['pred'])-base['candidate']))<1e-12;comparison=evidence(y,base['candidate'],pred,fold);assert comparison['eligible_for_new_partition_replication'];validation[stage]=comparison
        result=dict(status='passed',audited_utc=now(),main_plan_sha256=sha(ROOT/'research_r57_plan.json'),audit_plan_sha256=sha(ROOT/'research_r57_audit_plan.json'),verified_inner_tree_models=32,verified_inner_NN_models=8,verified_gate_models=8,model_records=records,validation=validation,rechecked_auxiliary_model_hashes=65,prior_source_audit_sha256=sha(ROOT/'research_r21_pre_inference_audit.json'),positive_class='True=1',peak_RSS_bytes=peak[0],CPU_threads=6,actual_top500_verified=False,limitation=plan['statistics_limitation']);dump(ROOT/'research_r57_pre_inference_audit.json',result);print('R57_INDEPENDENT_AUDIT_PASSED',flush=True)
    finally:stop.set();monitor.join(timeout=5)
if __name__=='__main__':main()

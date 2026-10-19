"""Independent CPU class-position/prediction replay and exact97field prefix proof."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,gc,threading,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_r4_common import ROOT,now,dump,sha
from research_r40_worker import watch_cpu
from research_r62_source_data import data,RATINGS,CONDITION,condition_columns
from research_r62_features import ClassLikelihoodFeatures
from research_r21_features import AuxiliaryFeatures
from research_r62_loader import auxiliary
from research_external_ablation import external
def main():
    plan=json.loads((ROOT/'research_r62_generator_audit_plan.json').read_text());sp=json.loads((ROOT/'research_r62_source_plan.json').read_text());assert plan['source_plan_sha256']==sha(ROOT/'research_r62_source_plan.json')
    for n,h in {**sp['source_sha256'],**sp['artifact_sha256'],**plan['input_sha256']}.items():assert sha(ROOT/n)==h,n
    stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/'research_r62_generator_audit_resource_failure.json'),daemon=True);watcher.start()
    try:
        source,sat,train,test,cols,scope=data();assert not scope['official_SAT_values_parsed'] and scope['all_rating_inputs_excluded'] and scope['source_SAT_used_as_conditioning_input'] and scope['source_rows']==129859;query=pd.concat([train,test],ignore_index=True);root=ROOT/'research_artifacts/r62_original_cohortclass';record=json.loads((root/'result.json').read_text());assert record['status']=='complete' and not record['official_SAT_values_parsed'] and len(record['models'])==13;matrices=[]
        for name,df in [('train',train),('test',test)]:
            z=np.load(root/f'{name}_features.npz');assert np.array_equal(z['id'],df.id) and record['features'][name]['sha256']==sha(root/f'{name}_features.npz') and list(z['columns'])==[f'original_cohort_class_llr_{j}' for j in range(13)];matrices.append(z['values'])
        values=np.concatenate(matrices);assert np.isfinite(values).all() and values.shape==(999479,13);checks=[]
        for j,rating in enumerate(RATINGS):
            run=root/f'rating{j}';r=json.loads((run/'result.json').read_text());assert r==record['models'][j] and r['status']=='complete' and r['source_rows']==129859 and r['source_positive_rows']==int(sat.sum()) and not r['official_SAT_values_parsed'] and r['source_SAT_conditioner'] and r['own_rating_excluded'] and r['parameters']==sp['parameters'];assert all(r[key]==sha(run/name) for key,name in [('model_sha256','model.cbm'),('likelihoods_sha256','likelihoods.npz'),('replay_sha256','replay.npz')]);assert r['peak_GPU_process_bytes']<=1073741824 and r['peak_RSS_bytes']<=8589934592
            z=np.load(run/'likelihoods.npz');assert np.array_equal(z['id'],query.id) and np.array_equal(z['llr'],values[:,j]);saved=np.load(run/'replay.npz');take=saved['idx'];assert np.array_equal(saved['id'],query.id.iloc[take]);features=condition_columns(cols);assert len(features)==8 and not set(features)&set(RATINGS) and r['all_rating_inputs_excluded'];assert r['feature_columns']==[*features,CONDITION] and rating not in r['feature_columns'] and 'id' not in r['feature_columns'] and 'satisfaction' not in r['feature_columns'];model=CatBoostClassifier(thread_count=6);model.load_model(str(run/'model.cbm'));classes=model.classes_.astype(int);assert model.feature_names_==r['feature_columns'] and model.tree_count_==300 and np.array_equal(classes,np.sort(source[rating].unique())) and classes.tolist()==r['classes'];observed=query[rating].to_numpy()[take].astype(int);pobs=[];errors=[]
            for hypothesis in [0,1]:
                Q=query.iloc[take][features].copy()
                for c in Q.select_dtypes(exclude='number'):Q[c]=Q[c].fillna('__MISSING__').astype(str)
                Q[CONDITION]=str(hypothesis)
                with threadpool_limits(6):p=model.predict_proba(Q,thread_count=6,task_type='CPU')
                assert p.shape==(len(take),len(classes)) and np.isfinite(p).all() and np.max(np.abs(p.sum(1)-1))<1e-6;column={int(value):i for i,value in enumerate(classes)};actual=np.array([p[i,column[value]] if value in column else 0. for i,value in enumerate(observed)]);error=float(np.max(np.abs(actual-saved[f'pobs{hypothesis}'])));assert error<1e-12;pobs.append(actual);errors.append(error)
            llr=(np.log(np.clip(pobs[1],1e-6,1))-np.log(np.clip(pobs[0],1e-6,1))).astype('float32');assert np.array_equal(llr,values[take,j]) and np.array_equal(llr,saved['llr']);checks.append(dict(rating_index=j,CPU_hypothesis_replay_errors=errors,actual_class_position_mapping_verified=True,model_sha256=r['model_sha256']));del model,p,z,saved;gc.collect()
        raw=train.drop(columns='id').iloc[:2048].reset_index(drop=True);y=np.arange(len(raw))%2;ext,_=external('both',train.id);ext=ext.iloc[:2048].reset_index(drop=True);aux=auxiliary(train.id).iloc[:2048].reset_index(drop=True);a=AuxiliaryFeatures().fit_transform(raw,y,ext,aux.iloc[:,:27]);b=ClassLikelihoodFeatures().fit_transform(raw,y,ext,aux);pd.testing.assert_frame_equal(a,b.iloc[:,:84],check_exact=True);assert b.shape==(2048,97) and np.array_equal(b.iloc[:,84:].to_numpy(),aux.iloc[:,27:].to_numpy('float32'))
        result=dict(status='passed_no_official_SAT_or_quality',audited_utc=now(),audit_plan_sha256=sha(ROOT/'research_r62_generator_audit_plan.json'),source_plan_sha256=sha(ROOT/'research_r62_source_plan.json'),source_manifest_sha256=sha(root/'result.json'),scope=scope,verified_source_models=13,CPU_replays=checks,semantic_proof=dict(rows=2048,labels='synthetic alternation only',prefix84_exact=True,appended13_LLRS_exact=True,feature_width=97),peak_RSS_bytes=peak[0],new_model_fits=0,official_quality_evaluated=False);dump(ROOT/'research_r62_generator_audit.json',result);print('R60_GENERATOR_AND97FIELD_AUDIT_PASSED',flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

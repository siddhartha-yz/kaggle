"""Thirteen original-only conditional rating densities; official queries use both SAT hypotheses."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'.cuda-cache-catboost-r62');Path(os.environ['CUDA_CACHE_PATH']).mkdir(exist_ok=True)
import json,gc,threading,time,resource,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_r4_common import now,dump,sha
from research_r23_guard import watch
from research_r62_source_data import data,frame,observed_probability,RATINGS,CONDITION,condition_columns

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    execution=json.loads((ROOT/'research_r62_source_build_plan.json').read_text());plan=json.loads((ROOT/'research_r62_source_plan.json').read_text());assert execution['source_plan_sha256']==sha(ROOT/'research_r62_source_plan.json') and execution['builder_sha256']==sha(__file__) and execution['source_audit_sha256']==sha(ROOT/'research_r62_source_audit.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    audited=json.loads((ROOT/'research_r62_source_audit.json').read_text());assert audited['status']=='passed_original_SAT_only_no_official_SAT' and not audited['scope']['official_SAT_values_parsed']
    out=ROOT/'research_artifacts/r62_original_cohortclass';out.mkdir(exist_ok=True);stop=threading.Event();peak=[0];watcher=threading.Thread(target=watch,args=(stop,peak,out/'resource_failure.json'),daemon=True);watcher.start();start=time.monotonic()
    try:
        source,source_sat,train,test,cols,scope=data();query=pd.concat([train,test],ignore_index=True);ids=query.id.to_numpy();assert len(query)==999479 and query.id.is_unique and 'satisfaction' not in query and not scope['official_SAT_values_parsed'];values=np.full((len(query),13),np.nan,'float32');models=[]
        record=dict(status='building',started_utc=now(),source_plan_sha256=sha(ROOT/'research_r62_source_plan.json'),build_plan_sha256=sha(ROOT/'research_r62_source_build_plan.json'),builder_sha256=sha(__file__),scope=scope,parameters=plan['parameters'],rating_nodes=RATINGS,models=[],official_SAT_values_parsed=False,external_original_SAT_conditioner=True,no_official_quality_scoring=True)
        for j,rating in enumerate(RATINGS):
            run=out/f'rating{j}';run.mkdir(exist_ok=True);result=run/'result.json';cache=run/'likelihoods.npz';modelpath=run/'model.cbm';features=condition_columns(cols);assert len(features)==8 and not set(features)&set(RATINGS) and 'satisfaction' not in features and 'id' not in features
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':
                r=json.loads(result.read_text());assert r['build_plan_sha256']==sha(ROOT/'research_r62_source_build_plan.json') and r['model_sha256']==sha(modelpath) and r['likelihoods_sha256']==sha(cache) and r['replay_sha256']==sha(run/'replay.npz');z=np.load(cache);assert np.array_equal(z['id'],ids);values[:,j]=z['llr'];models.append(r);continue
            tick=time.monotonic();dump(ROOT/'research_r62_source_state.json',dict(status='source_rating_fit',rating_index=j,rating=rating,pid=os.getpid(),active_training=True,updated_utc=now()));X=frame(source,features,source_sat);cats=X.select_dtypes(exclude='number').columns.tolist();assert CONDITION in cats and X.shape==(129859,9);target=source[rating].to_numpy();assert np.isfinite(target).all()
            with threadpool_limits(6):
                model=CatBoostClassifier(**plan['parameters']);model.fit(X,target,cat_features=cats);assert model.tree_count_==300 and model.get_all_params()['task_type']=='GPU' and np.array_equal(model.classes_,np.sort(np.unique(target))) and model.feature_names_==list(X);model.save_model(str(modelpath));loaded=CatBoostClassifier(thread_count=6);loaded.load_model(str(modelpath));pobs=np.empty((len(query),2),'float64');unknown=[0,0];batch=plan['official_query_CPU_batch_rows']
                for hypothesis in range(2):
                    for lo in range(0,len(query),batch):
                        hi=min(lo+batch,len(query));Q=frame(query.iloc[lo:hi],features,str(hypothesis));p,missing=observed_probability(model,Q,query[rating].to_numpy()[lo:hi]);pobs[lo:hi,hypothesis]=p;unknown[hypothesis]+=missing
                    dump(ROOT/'research_r62_source_state.json',dict(status='source_rating_inference',rating_index=j,hypothesis=hypothesis,pid=os.getpid(),active_training=False,updated_utc=now()))
                llr=(np.log(np.clip(pobs[:,1],1e-6,1))-np.log(np.clip(pobs[:,0],1e-6,1))).astype('float32');assert np.isfinite(llr).all();take=np.sort(np.random.default_rng(20261060+j).choice(len(query),4096,replace=False));replays=[]
                for hypothesis in range(2):
                    Q=frame(query.iloc[take],features,str(hypothesis));p,_=observed_probability(loaded,Q,query[rating].to_numpy()[take]);error=float(np.max(np.abs(p-pobs[take,hypothesis])));assert error<1e-12;replays.append(error)
            np.savez_compressed(cache,id=ids,llr=llr);np.savez_compressed(run/'replay.npz',idx=take,id=ids[take],pobs0=pobs[take,0],pobs1=pobs[take,1],llr=llr[take]);r=dict(status='complete',completed_utc=now(),rating_index=j,rating=rating,source_plan_sha256=sha(ROOT/'research_r62_source_plan.json'),build_plan_sha256=sha(ROOT/'research_r62_source_build_plan.json'),builder_sha256=sha(__file__),feature_columns=list(X),categorical_columns=cats,classes=model.classes_.tolist(),source_rows=len(source),own_rating_excluded=True,all_rating_inputs_excluded=True,official_SAT_values_parsed=False,source_SAT_conditioner=True,source_positive_rows=int(source_sat.sum()),parameters=plan['parameters'],resolved_parameters=model.get_all_params(),model_sha256=sha(modelpath),likelihoods_sha256=sha(cache),replay_sha256=sha(run/'replay.npz'),CPU_reload_errors=replays,unseen_official_rating_class_counts=unknown,likelihood_ratio_std=float(llr.std()),peak_GPU_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-tick);assert 0<r['peak_GPU_process_bytes']<=1073741824 and r['peak_RSS_bytes']<=8589934592;dump(result,r);models.append(r);values[:,j]=llr;record['models']=models;dump(out/'result.json',record);print('SOURCE_NODE_COMPLETE',j,rating,r['seconds'],r['peak_GPU_process_bytes'],flush=True);del model,loaded,X,Q,pobs,llr;gc.collect()
        assert len(models)==13 and np.isfinite(values).all() and np.std(values)>1e-6;feature_records={}
        for name,lo,hi in [('train',0,len(train)),('test',len(train),len(query))]:
            path=out/f'{name}_features.npz';np.savez_compressed(path,id=ids[lo:hi],values=values[lo:hi],columns=np.array([f'original_cohort_class_llr_{j}' for j in range(13)]));feature_records[name]=dict(rows=hi-lo,sha256=sha(path))
        record.update(status='complete',completed_utc=now(),models=models,features=feature_records,peak_GPU_process_bytes=peak[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start);dump(out/'result.json',record);dump(ROOT/'research_r62_source_result.json',record);dump(ROOT/'research_r62_source_state.json',dict(status='complete',updated_utc=now(),active_training=False));print('R62_SOURCE_FEATURES_COMPLETE_NO_OFFICIAL_SAT',flush=True)
    finally:stop.set();watcher.join(timeout=5)
if __name__=='__main__':main()

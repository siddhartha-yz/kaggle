"""Independent source provenance/exact-match/label-scope audit, separate from quality gates."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,gc,hashlib,joblib,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def canonical(frame,columns):
    out=frame.loc[:,columns].copy()
    for c in columns:
        if pd.api.types.is_numeric_dtype(out[c]):out[c]=pd.to_numeric(out[c]).astype(np.float64)
        else:out[c]=out[c].fillna('__MISSING__').map(str).str.strip().str.casefold()
    return out

def main():
    audit_plan=json.loads((ROOT/'research_r45_source_audit_plan.json').read_text())
    for n,h in audit_plan['input_sha256'].items():assert sha(ROOT/n)==h,n
    assert audit_plan['audit_source_sha256']==sha(__file__)
    R=ROOT/'research_sources/invistico_candidate';meta=json.loads((R/'public_metadata.json').read_text());fresh=json.loads((ROOT/'research_r45_source_metadata_recheck.json').read_text());download=json.loads((R/'download.json').read_text());inspection=json.loads((R/'inspection.json').read_text());plan=json.loads((ROOT/'research_r44_plan.json').read_text());out=ROOT/'research_artifacts/r44_invistico_teacher';teacher=json.loads((out/'result.json').read_text());selection=json.loads((out/'selection.json').read_text())
    assert meta['ref']==fresh['ref']==download['ref']=='sjleshrac/airlines-customer-satisfaction' and meta['licenseName']==fresh['licenseName']==download['license']=='CC0: Public Domain'
    assert meta['currentVersionNumber']==fresh['currentVersionNumber']==download['dataset_version']==1 and download['csv_bytes']==12612566
    assert sha(R/'Invistico_Airline.csv')==download['csv_sha256']=='584f8e338d189117f1c8f8c2ee8fb0b5d1d6d23ff5df67417272e8ee360da7a2' and sha(R/'official_version1.zip')==download['zip_sha256']
    last=datetime.fromisoformat(fresh['lastUpdated'].replace('Z','+00:00'));assert last<datetime(2026,10,1,tzinfo=timezone.utc)
    assert datetime.fromisoformat(meta['checked_utc'])<datetime.fromisoformat(plan['frozen_utc'])<datetime.fromisoformat(teacher['completed_utc'])
    # Guard actual dataframe loads: official SAT values may never be parsed here.
    real_read=pd.read_csv;events=[];official_paths={(ROOT/'data/train.csv').resolve(),(ROOT/'data/test.csv').resolve()}
    def guarded_read(path,*args,**kwargs):
        p=Path(path).resolve()
        if p in official_paths:
            if kwargs.get('nrows')==0:events.append(dict(file=p.name,kind='header_only'))
            else:
                used=kwargs.get('usecols');assert isinstance(used,list) and 'satisfaction' not in used;events.append(dict(file=p.name,kind='explicit_covariates_only',column_count=len(used)))
        return real_read(path,*args,**kwargs)
    pd.read_csv=guarded_read
    try:
        columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0).columns if c not in ['id','satisfaction']];assert len(columns)==21
        raw=pd.read_csv(R/'Invistico_Airline.csv');assert len(raw)==129880 and set(raw)==set(columns+['satisfaction','Online support'])
        official={name:pd.read_csv(ROOT/f'data/{name}.csv',usecols=['id',*columns]) for name in ['train','test']}
        source=canonical(raw,columns);queries=pd.concat([canonical(official['train'],columns),canonical(official['test'],columns)],ignore_index=True);source_hash=pd.util.hash_pandas_object(source,index=False).to_numpy();query_hash=pd.util.hash_pandas_object(queries,index=False).to_numpy();keep=~np.isin(source_hash,query_hash);saved=np.load(R/'safe_mask.npy');assert np.array_equal(keep,saved) and keep.sum()==129878 and (~keep).sum()==2 and not np.isin(source_hash[keep],query_hash).any()
        matched_pairs=0
        for i in np.flatnonzero(~keep):
            hits=np.flatnonzero(query_hash==source_hash[i]);verified=[j for j in hits if source.iloc[i].equals(queries.iloc[j])];assert verified;matched_pairs+=len(verified)
        assert matched_pairs==2 and sha(R/'safe_mask.npy')==inspection['new_source_safe_mask_sha256']
        labels=raw.satisfaction.map(str).str.strip().str.casefold();assert set(labels)=={'satisfied','dissatisfied'};source_y=(labels.to_numpy()=='satisfied').astype(np.int8)[keep];safe=source.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        # Independently built rows must also match the guarded historical data loader.
        from research_r44_source import source_data
        helper_x,helper_y,_,_,helper_columns,helper_provenance=source_data();pd.testing.assert_frame_equal(helper_x,safe,check_exact=True);assert np.array_equal(helper_y,source_y) and helper_columns==columns and helper_provenance['excluded_match_pairs_verified_value_by_value']==2
        a,b=train_test_split(np.arange(len(safe)),test_size=.2,stratify=source_y,random_state=20261018);split=np.load(out/'source_split.npz');assert np.array_equal(a,split['fit_idx']) and np.array_equal(b,split['valid_idx']) and not np.intersect1d(a,b).size
        assert teacher['status']=='complete' and teacher['provenance']['safe_rows']==129878 and not teacher['official_satisfaction_loaded'] and teacher['teacher_selected_on_source_only'] and not teacher['source_extra_Online_support_used'] and teacher['raw_source_features']==columns
        assert teacher['plan_sha256']==sha(ROOT/'research_r44_plan.json') and teacher['source_selection_sha256']==sha(out/'selection.json')
        selected=teacher['selected_trees'];curve=selection['history']['valid_0']['auc'];assert selected==selection['selected_trees']==int(np.argmax(curve))+1 and len(curve)<=3000 and not selection['official_targets_used']
        for key,file in [('source_split_sha256','source_split.npz'),('selection_model_sha256','selection_model.txt'),('selection_transformer_sha256','selection_transformer.joblib')]:assert selection[key]==sha(out/file)
        inner=joblib.load(out/'selection_transformer.joblib');full=joblib.load(out/'transformer.joblib');assert inner.n==len(a) and full.n==len(safe) and inner.columns==full.columns==columns and inner.mode==full.mode=='base' and not inner.te_maps and not full.te_maps
        for tr,idx in [(inner,a),(full,np.arange(len(safe)))]:
            pd.testing.assert_series_equal(tr.counts.sort_index(),safe.iloc[idx].groupby('Flight Distance',sort=False).size().sort_index())
            assert 'satisfaction' not in tr.columns and 'Online support' not in tr.columns and 'id' not in tr.columns and len(tr.output_columns)==25
        selected_model=lgb.Booster(model_file=str(out/'selection_model.txt'));model=lgb.Booster(model_file=str(out/'model.txt'));assert model.num_trees()==selected_model.num_trees()==selected and model.num_feature()==selected_model.num_feature()==25 and sha(out/'model.txt')==teacher['model_sha256'] and sha(out/'transformer.joblib')==teacher['transformer_sha256']
        with threadpool_limits(6):
            V=inner.transform_te(safe.iloc[b].reset_index(drop=True));pred=selected_model.predict(V,num_threads=6);auc_error=abs(float(roc_auc_score(source_y[b],pred))-selection['source_AUC']);assert auc_error<1e-12 and abs(selection['source_AUC']-curve[selected-1])<1e-12
            replays={}
            for j,name in enumerate(['train','test']):
                frame=official[name];cache=np.load(out/f'{name}_features.npz');assert np.array_equal(cache['id'],frame.id) and list(cache['columns'])==['invistico_teacher_lgb'] and cache['values'].shape==(len(frame),1) and sha(out/f'{name}_features.npz')==teacher['features'][name]['features_sha256'];assert np.isfinite(cache['values']).all() and ((cache['values']>=0)&(cache['values']<=1)).all()
                take=np.random.default_rng(20261045+j).choice(len(frame),4096,replace=False);V=full.transform_te(canonical(frame.iloc[take].reset_index(drop=True),columns));assert model.feature_name()==list(V);replay=model.predict(V,num_threads=6);error=float(np.max(np.abs(replay-cache['values'][take,0])));assert error<1e-12;replays[name]=dict(rows=4096,CPU_replay_error=error)
    finally:pd.read_csv=real_read
    result=dict(status='passed_source_provenance_and_exactmatch_scope_only',audited_utc=datetime.now(timezone.utc).isoformat(),source_ref=fresh['ref'],source_page='https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction',public_metadata_url=fresh['public_metadata_url'],first_party_license=fresh['licenseName'],source_version=1,first_party_last_updated=fresh['lastUpdated'],source_predates_competition=True,source_csv_sha256=download['csv_sha256'],source_rows=129880,safe_source_rows=129878,globally_excluded_source_rows=2,excluded_match_pairs_value_verified=matched_pairs,remaining_exact_official_covariate_matches=0,official_covariate_rows=999479,match_features=columns,match_normalization='numericfloat64/textstripcasefold;all21availablepredictors;id/targetsnotincluded',official_target_values_parsed=False,official_targets_used_for_source_fit_or_selection=False,actual_guarded_official_reads=events,source_only_selection_split_verified=True,source_only_transformer_training_row_counts_verified=True,source_only_selected_trees=selected,source_class_mapping='satisfied=1, dissatisfied=0',source_selection_AUC_replay_error=auc_error,teacher_feature_replays=replays,Online_support_excluded_from_teacher_inputs=True,source_choice_basis='PublicCC0versionedrelatedsatisfactiontask,21sharedpredictorsanddistinctpubliclabelledtable;chosenandfrozenbeforeanyR44officialCVevaluation,notbyrankingdatasetsontheofficialvalidationlabels.',source_choice_chronology=dict(metadata_checked=meta['checked_utc'],covariates_inspected=inspection['checked_utc'],R44_plan_frozen=plan['frozen_utc'],source_teacher_completed=teacher['completed_utc']),important_limitations=['Two21-featuretableshavingonly2exactsharedrowsdoesNOTproveindependentrespondentsorgenerationprocess;theycanberelated/modifiedversions.Noindependent-samplingclaim.','Newsourcecovariatealignmentisweaker(nearkey8.4%vsold37%);notassertedtruegenerator.','R44officialfivefoldfailedconfidencegates;R45neuralfollow-upwaschosenafterthatresult,soofficialvalidationhistoryisreusedandCIsremainconditional.','Thisauditdoesnotqualifyapredictionorclaimrank;5+3qualitygatesandstudentmodelauditsarestillrequired.'],audit_plan_sha256=sha(ROOT/'research_r45_source_audit_plan.json'))
    dump(ROOT/'research_r45_source_audit.json',result);print('SOURCE_AUDIT_PASS',result['safe_source_rows'],result['globally_excluded_source_rows'],replays,flush=True)
if __name__=='__main__':main()

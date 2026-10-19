"""Compress cached65-model posterior outputs without SAT loading or auxiliary fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
from research_r4_common import ROOT,now,dump,sha,guard
from research_r19_aux_source import RATINGS
from research_r37_rank_statistics import conditional_normal_rank
import json,numpy as np,pandas as pd,resource

def main():
    plan=json.loads((ROOT/'research_r37_plan.json').read_text())
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256']}.items():assert sha(ROOT/n)==h,n
    src=ROOT/'research_artifacts/r25_posterior_officialcb';prior=json.loads((src/'result.json').read_text());assert prior['status']=='complete' and prior['source_models_used']==65 and not prior['satisfaction_loaded'] and prior['no_new_models_fitted']
    models=ROOT/'research_artifacts/r20_aux_officialcb';source=json.loads((models/'result.json').read_text());assert not source['satisfaction_column_loaded'] and not source['satisfaction_used_as_target_or_input']
    for row in source['models']:assert sha(models/f"rating{row['rating_index']}_fold{row['auxiliary_fold']}/model.cbm")==row['model_sha256']
    out=ROOT/'research_artifacts/r37_ordinal_ranks';out.mkdir(exist_ok=False);record=dict(status='generating',started_utc=now(),plan_sha256=sha(ROOT/'research_r37_plan.json'),posterior_manifest_sha256=sha(src/'result.json'),satisfaction_column_loaded=False,new_auxiliary_models_fitted=0,source_models_reused=65,transductive=True,feature_columns=[f'conditional_positive_rank_{j}' for j in range(13)],zero_state_handling='state0neutralrank0,inductivebiasseparatefromordered1..5')
    for name in ['train','test']:
        guard();df=pd.read_csv(ROOT/f'data/{name}.csv',usecols=['id',*RATINGS]);z=np.load(src/f'{name}_features.npz');assert np.array_equal(z['id'],df.id) and sha(src/f'{name}_features.npz')==prior[f'{name}_features_sha256'];assert list(z['columns'])==[f'rating_posterior_{j}_class{k}' for j in range(13) for k in range(6)]
        probabilities=z['values'];values=np.empty((len(df),13),'float32')
        for j,c in enumerate(RATINGS):values[:,j]=conditional_normal_rank(probabilities[:,6*j:6*(j+1)],np.arange(6),df[c].to_numpy())
        assert np.isfinite(values).all();np.savez_compressed(out/f'{name}_features.npz',id=df.id.to_numpy(),values=values,columns=np.asarray(record['feature_columns']));record[f'{name}_features_sha256']=sha(out/f'{name}_features.npz');record[f'{name}_rows']=len(df);print('RANKS',name,values.shape,flush=True)
    record.update(status='complete',completed_utc=now(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(out/'result.json',record)
if __name__=='__main__':main()

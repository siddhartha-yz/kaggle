"""Covariate-only numerical scale diagnosis and exact two-coordinate representation proof."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,numpy as np,pandas as pd
from research_r4_common import ROOT,now,dump,sha
from research_r23_loader import auxiliary
from research_r23_features import ConditionalFeatures
from research_r59_features import DelayLogFeatures,DELAYS
def main():
    profiles={};train=None
    for name in ['train','test']:
        cols=[c for c in pd.read_csv(ROOT/f'data/{name}.csv',nrows=0) if c!='satisfaction'];df=pd.read_csv(ROOT/f'data/{name}.csv',usecols=cols);profiles[name]={}
        for c in DELAYS:
            value=df[c].fillna(0).to_numpy('float32');assert np.isfinite(value).all() and (value>=0).all();q=np.quantile(value,[0,.25,.5,.75,.95,.99,1]);assert q[3]-q[1]==0 and q[-1]>0
            profiles[name][c]=dict(missing=int(df[c].isna().sum()),nonzero=int(np.count_nonzero(value)),quantiles=dict(zip(['min','q25','median','q75','q95','q99','max'],q.tolist())),IQR=0.,raw_range_fallback_scale=float(2/(value.max()-value.min())),raw_preclip_q95=float(2*q[4]/q[-1]),log_preclip_q95=float(2*np.log1p(q[4])/np.log1p(q[-1])))
        if name=='train':train=df
    raw=train.drop(columns='id').iloc[:2048].reset_index(drop=True);y=np.arange(len(raw))%2;aux=auxiliary(train.id).iloc[:2048].reset_index(drop=True);a=ConditionalFeatures().fit_transform(raw,y,aux);b=DelayLogFeatures().fit_transform(raw,y,aux);pd.testing.assert_frame_equal(a.drop(columns=DELAYS),b.drop(columns=DELAYS),check_exact=True)
    for c in DELAYS:
        assert np.array_equal(b[c].to_numpy(),np.log1p(a[c].to_numpy('float64')).astype('float32'));assert np.unique(a[c]).size==np.unique(b[c]).size
    assert b.shape==(2048,74) and b.select_dtypes(include='number').shape[1]==52
    pipeline=ROOT.parent/'.venv-gpu/lib/python3.12/site-packages/pytabkit/models/nn_models/pipeline.py'
    result=dict(status='passed_no_official_SAT_or_quality',checked_utc=now(),profiles=profiles,implementation=dict(package='pytabkit1.7.3',path=str(pipeline.relative_to(ROOT.parent)),sha256=sha(pipeline),class_name='RobustScaleFactory',zero_IQR_fallback='quant_diff=.5*(max-min); scale=1/(quant_diff+1e-30)',interpretation='Both delays have zero interquartile range; the min/max fallback places q95 near.03 before smoothclip. Fixed log1p raises the same q95 near.67, spreading the common nonzero delays while retaining original categorical twins. This is finite-network preconditioning, not a bug correction or a gain claim.'),semantic_proof=dict(rows=2048,labels='synthetic alternation only',official_SAT_values_parsed=False,changed_numeric_coordinates=2,unchanged_fields_exact=72,raw_category_identities_unchanged=True,feature_width=74,numeric_fields=52,nominal_fields=22),source_sha256={n:sha(ROOT/n) for n in ['research_r59_features.py','research_r59_input_audit.py']});dump(ROOT/'research_r59_input_audit.json',result);print('R59_COVARIATE_SCALE_AND_VIEW_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()

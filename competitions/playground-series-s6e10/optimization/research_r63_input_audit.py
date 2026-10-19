"""Synthetic labels only: exact61 retained coordinates and13 source likelihood ratios."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='6'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,pandas as pd,json
from research_r4_common import ROOT,now,dump,sha
from research_r23_features import ConditionalFeatures
from research_r23_loader import auxiliary as oldaux
from research_r63_loader import auxiliary
def main():
    g=json.loads((ROOT/'research_r62_generator_audit.json').read_text());assert g['status']=='passed_no_official_SAT_or_quality' and g['verified_source_models']==13 and not g['scope']['official_SAT_values_parsed'];cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0) if c!='satisfaction'];df=pd.read_csv(ROOT/'data/train.csv',usecols=cols);raw=df.drop(columns='id').iloc[:2048].reset_index(drop=True);y=np.arange(len(raw))%2;old=oldaux(df.id).iloc[:2048].reset_index(drop=True);new=auxiliary(df.id).iloc[:2048].reset_index(drop=True);a=ConditionalFeatures().fit_transform(raw,y,old);b=ConditionalFeatures().fit_transform(raw,y,new);pd.testing.assert_frame_equal(a.iloc[:,:47],b.iloc[:,:47],check_exact=True);pd.testing.assert_frame_equal(a.iloc[:,60:],b.iloc[:,60:],check_exact=True);assert b.shape==(2048,74) and b.select_dtypes(include='number').shape[1]==52 and np.array_equal(b.iloc[:,47:60].to_numpy(),new.iloc[:,:13].to_numpy('float32'))
    r=dict(status='passed_synthetic_labels_no_official_SAT_or_quality',audited_utc=now(),generator_audit_sha256=sha(ROOT/'research_r62_generator_audit.json'),proof=dict(rows=2048,unchanged_fields_exact=61,replaced_numeric_fields=13,feature_width=74,numeric_fields=52,nominal_fields=22,raw_category_identities_unchanged=True,old14surprisals_unchanged=True,original_raw_delay_coordinates_unchanged=True),official_SAT_values_parsed=False,labels='synthetic alternation only',new_model_fits=0);dump(ROOT/'research_r63_input_audit.json',r);print('R61_NEURAL_VIEW_PROOF_PASSED',flush=True)
if __name__=='__main__':main()

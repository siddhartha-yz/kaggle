"""DistinctpublicCC0Invisticovariant; sourceonlylabels, globalofficialXexclusion."""
from research_r4_common import ROOT,sha
import json,numpy as np,pandas as pd
SRC=ROOT/'research_sources/invistico_candidate'

def normalize(frame,columns):
    out=frame[columns].copy()
    for c in columns:
        if pd.api.types.is_numeric_dtype(out[c]):out[c]=out[c].astype('float64')
        else:out[c]=out[c].fillna('__MISSING__').astype(str).str.strip().str.casefold()
    return out

def source_data():
    metadata=json.loads((SRC/'public_metadata.json').read_text());download=json.loads((SRC/'download.json').read_text());inspection=json.loads((SRC/'inspection.json').read_text())
    assert metadata['licenseName']=='CC0: Public Domain' and metadata['currentVersionNumber']==1 and download['csv_sha256']=='584f8e338d189117f1c8f8c2ee8fb0b5d1d6d23ff5df67417272e8ee360da7a2'
    assert sha(SRC/'Invistico_Airline.csv')==download['csv_sha256']
    raw=pd.read_csv(SRC/'Invistico_Airline.csv');columns=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0) if c not in ['id','satisfaction']]
    assert len(raw)==129880 and set(raw)==set(columns+['satisfaction','Online support'])
    train=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*columns]);test=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*columns]);A=normalize(raw,columns);B=pd.concat([normalize(train,columns),normalize(test,columns)],ignore_index=True)
    a=pd.util.hash_pandas_object(A,index=False).to_numpy();b=pd.util.hash_pandas_object(B,index=False).to_numpy();keep=~np.isin(a,b);saved=np.load(SRC/'safe_mask.npy');assert np.array_equal(keep,saved) and keep.sum()==129878 and sha(SRC/'safe_mask.npy')==inspection['new_source_safe_mask_sha256']
    exact_pairs=0
    for i in np.flatnonzero(~keep):
        candidates=np.flatnonzero(b==a[i]);matches=[j for j in candidates if A.iloc[i].equals(B.iloc[j])];assert matches;exact_pairs+=len(matches)
    labels=raw.satisfaction.astype(str).str.strip().str.casefold();assert set(labels)=={'satisfied','dissatisfied'};y=(labels=='satisfied').to_numpy('int8')[keep]
    return A.loc[keep].reset_index(drop=True),y,train,test,columns,dict(safe_rows=int(keep.sum()),excluded_source_rows=int((~keep).sum()),excluded_match_pairs_verified_value_by_value=exact_pairs,source_labels_only=True,official_SAT_column_loaded=False,unobserved_Online_support_excluded=True,public_CC0_source=True,source_ref=metadata['ref'],source_version=1)

"""Independent original-only class conditioner; official files expose covariates only."""
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
CONDITION='__original_SAT_hypothesis__'
def canonical(frame,cols):
    out=frame[cols].copy()
    for c in out:
        if pd.api.types.is_numeric_dtype(out[c]):out[c]=out[c].astype('float64')
        else:out[c]=out[c].fillna('__MISSING__').astype(str)
    return out
def data():
    events=[];official={(ROOT/f'data/{n}.csv').resolve() for n in ['train','test']};original_read=pd.read_csv
    def read(path,*args,**kwargs):
        if Path(path).resolve() in official:
            assert kwargs.get('nrows')==0 or (isinstance(kwargs.get('usecols'),list) and 'satisfaction' not in kwargs['usecols'])
            events.append(dict(file=Path(path).name,header_only=kwargs.get('nrows')==0,columns=kwargs.get('usecols')))
        return original_read(path,*args,**kwargs)
    pd.read_csv=read
    try:
        cols=[c for c in pd.read_csv(ROOT/'data/train.csv',nrows=0) if c not in ['id','satisfaction']];train=pd.read_csv(ROOT/'data/train.csv',usecols=['id',*cols]);test=pd.read_csv(ROOT/'data/test.csv',usecols=['id',*cols]);source=pd.read_csv(ROOT/'research_sources/original_cc0/data.csv')
    finally:pd.read_csv=original_read
    assert len(cols)==21 and len(train)==699635 and len(test)==299844 and len(source)==129880 and source.satisfaction.dtype==bool and set(source)==set([*cols,'satisfaction']) and all(c in cols for c in RATINGS)
    A=canonical(source,cols);B=canonical(pd.concat([train[cols],test[cols]],ignore_index=True),cols);ha=pd.util.hash_pandas_object(A,index=False).to_numpy();hb=pd.util.hash_pandas_object(B,index=False).to_numpy();keep=~np.isin(ha,hb);saved=np.load(ROOT/'research_sources/original_safe_mask.npy');assert np.array_equal(keep,saved) and keep.sum()==129859 and (~keep).sum()==21
    for i in np.flatnonzero(~keep):
        matches=np.flatnonzero(hb==ha[i]);assert any(A.iloc[[i]].reset_index(drop=True).equals(B.iloc[[j]].reset_index(drop=True)) for j in matches)
    assert not np.isin(ha[keep],hb).any();safe=source.loc[keep].reset_index(drop=True);audit=dict(source_rows=len(safe),excluded_exact_source_rows=21,excluded_pairs_value_verified=True,official_SAT_values_parsed=False,source_SAT_values_parsed=True,source_SAT_used_as_conditioning_input=True,source_SAT_used_as_rating_target=False,own_rating_excluded_from_model_inputs=True,known_covariate_rows=len(B),guarded_official_reads=events)
    return safe[cols],safe.satisfaction.to_numpy('int8'),train,test,cols,audit
def frame(raw,features,hypothesis):
    out=raw[features].copy()
    for c in out.select_dtypes(exclude='number'):out[c]=out[c].fillna('__MISSING__').astype(str)
    out[CONDITION]=hypothesis if isinstance(hypothesis,str) else np.asarray(hypothesis,dtype='int8').astype(str)
    assert 'satisfaction' not in out and 'id' not in out
    return out
def observed_probability(model,query,observed):
    classes=np.asarray(model.classes_,dtype=int);assert np.array_equal(classes,np.unique(classes));p=model.predict_proba(query,thread_count=6,task_type='CPU');assert p.shape==(len(query),len(classes)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(1)-1))<1e-6
    position=np.searchsorted(classes,np.asarray(observed,dtype=int));known=position<len(classes);known[known]=classes[position[known]]==np.asarray(observed,dtype=int)[known];out=np.zeros(len(query));out[known]=p[np.flatnonzero(known),position[known]]
    return out,int((~known).sum())

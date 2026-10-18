"""Covariate-only diagnostic; no satisfaction values are accessed by the matching/null comparison."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json
from datetime import datetime,timezone
from research_original import safe_original
R=Path(__file__).resolve().parent
source,train,test,cols,audit=safe_original();source=source[cols];raw=train[cols];rng=np.random.default_rng(20261026);idx=np.sort(rng.choice(len(raw),5000,replace=False));query=raw.iloc[idx].reset_index(drop=True);null=query.copy();mutate=[c for c in cols if c not in ['Flight Distance','Age','Gender','Class','Type of Travel','Customer Type']];keys=['Flight Distance','Class','Type of Travel','Customer Type'];groups=raw.groupby(keys,sort=False).indices
for i,row in query.iterrows():
 ids=groups[tuple(row[k] for k in keys)]
 for c in mutate:null.at[i,c]=raw[c].iloc[rng.choice(ids)]
def canon(d):
 x=d.copy()
 for c in x:x[c]=x[c].astype('float64').astype('string').fillna('__MISSING__') if pd.api.types.is_numeric_dtype(x[c]) else x[c].fillna('__MISSING__').astype(str)
 return x
A=canon(source);B=canon(query);C=canon(null);fields=[c for c in cols if c!='Flight Distance'];origgroups=source.groupby('Flight Distance').indices;real=[];fake=[];norig=[]
for i,row in query.iterrows():
 ids=origgroups.get(row['Flight Distance'])
 if ids is None:continue
 a=A.iloc[ids][fields].to_numpy();real.append(int(np.max((a==B.iloc[i][fields].to_numpy()).sum(axis=1))));fake.append(int(np.max((a==C.iloc[i][fields].to_numpy()).sum(axis=1))));norig.append(len(ids))
real=np.asarray(real);fake=np.asarray(fake);r={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'descriptive_covariate_diagnostic_only','official_satisfaction_used':False,'source_satisfaction_used':False,'query_rows':5000,'matched_FD_rows':len(real),'null':'KeepsFD/Age/Gender/Class/Travel/Customerfixed; independentlyresamplesratingsanddelaysfromofficialpredictorswithinFD/Class/Travel/Customerstrata. NoSATstratification.','real_mean_max_exact_fields_of20':float(real.mean()),'null_mean_max_exact_fields_of20':float(fake.mean()),'paired_mean_difference':float(np.mean(real-fake)),'real_hist':np.bincount(real,minlength=21).tolist(),'null_hist':np.bincount(fake,minlength=21).tolist(),'median_source_candidates_perFD':float(np.median(norig)),'caution':'Jointretentiondoesnotproveexemplarcopying, recoverableoriginlabels orcausality; groupmarginalindependencenull isdeliberatelysimple. Onlygrounds whetherprobabilisticlinkageisworthonefrozentrial.'};(R/'research_copy_structure_diagnostic.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

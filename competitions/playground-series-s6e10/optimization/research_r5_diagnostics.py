"""Exploratory reused-label diagnostics; no model fitting or significance-based subgroup claims."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
D=pd.read_csv(R/'data/train.csv');y=D.satisfaction.to_numpy('int8');z=np.load(R/'research_artifacts/r4_lean_gpu_init_oof/oof.npz');old=np.load(R/'research_artifacts/gpu_value_raw_oof_n2/oof.npz');assert np.array_equal(z['id'],D.id) and np.array_equal(z['target'],y);p=z['candidate'];q=old['candidate'];fold=z['fold'];pairs=json.loads((R/'research_artifacts/r4_rating_relations_oof/selection.json').read_text())['chosen'];equal=np.mean([(D[a['columns'][0]].to_numpy()==D[a['columns'][1]].to_numpy()).astype('float32') for a in pairs],axis=0);coherence=pd.cut(equal,[-.01,.25,.5,.75,1.01],labels=['low','midlow','midhigh','high']);segment=D['Class']+'|'+D['Type of Travel']+'|'+D['Customer Type'];masks={}
for key,values in [('segment',segment),('coherence',coherence),('segment_coherence',segment.astype(str)+'|'+coherence.astype(str))]:
 for value in sorted(pd.Series(values).unique()):masks[key+':'+str(value)]=np.asarray(values==value)
rows=[]
for name,m in masks.items():
 if m.sum()<1500 or np.unique(y[m]).size<2:continue
 residual=y[m]-p[m];se=float(np.std(residual,ddof=1)/np.sqrt(m.sum()));by=[]
 for k in range(5):
  a=m&(fold==k);by.append({'n':int(a.sum()),'calibration_residual':float(np.mean(y[a]-p[a]))})
 rows.append({'slice':name,'rows':int(m.sum()),'positive_rate':float(y[m].mean()),'mean_v7':float(p[m].mean()),'calibration_residual':float(residual.mean()),'descriptive_residual_se':se,'v7_auc':float(roc_auc_score(y[m],p[m])),'v6_auc':float(roc_auc_score(y[m],q[m])),'v7_logloss':float(log_loss(y[m],p[m])),'fold_calibration':by})
record={'created_utc':datetime.now(timezone.utc).isoformat(),'label_status':'Repeatedlyusedofficiallabels; exploratorydiagnostic only, no untouchedholdout or multiplicity-corrected subgroupclaim','mechanisms':'source-selected12ratingpaircoherence and fixedClass/Travel/Customersegments. No thresholdsearch.','slices':rows,'next_prepared_route':'34autoTEresponses, raw21 +13rating xsegment; keys preparedbeforev7publicresult.'};(R/'research_r5_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps([x for x in rows if not x['slice'].startswith('segment_coherence')],indent=2))

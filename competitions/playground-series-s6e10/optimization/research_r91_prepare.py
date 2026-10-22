import json,joblib,pandas as pd,numpy as np,shutil
from research_r91_common import ROOT,sha,dump,now,refs
from research_r91_features import MaxSurprisal,SURPRISALS,FIELD
from research_external_ablation import external
from research_r23_loader import auxiliary
from research_oof import params
from research_resume_policy_20261021 import check_training
check_training();assert not (ROOT/'research_r91_plan.json').exists() and not (ROOT/'research_artifacts/r91_max_surprisal').exists()
free=shutil.disk_usage(ROOT).free;assert free>=2*2**30+256*2**20+2**20,'Reserve2GiBplusentire256MiBoutputbudget'
f=pd.read_csv(ROOT/'data/train.csv');raw=f.drop(columns=['id','satisfaction']);ext,_=external('both',f.id);aux=auxiliary(f.id);z=refs('fivefold');vi=np.flatnonzero(z[0]['fold']==0)
control=joblib.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_fivefold/fold0/transformer.joblib');tr=MaxSurprisal();tr.base=control;tr.columns=control.columns+[FIELD]
q=raw.iloc[vi].reset_index(drop=True);e=ext.iloc[vi].reset_index(drop=True);a=aux.iloc[vi].reset_index(drop=True);V=tr.transform(q,e,a);C=control.transform(q,e,a);pd.testing.assert_frame_equal(V.iloc[:,:84],C,check_exact=True)
assert np.array_equal(V[FIELD],a[SURPRISALS].max(axis=1).to_numpy('float32')) and V[FIELD].nunique()>1
example=np.array([[13]+[0]*12,[1]*13],dtype='float32');assert example.mean(axis=1).tolist()==[1,1] and example.max(axis=1).tolist()==[13,1]
old=json.loads((ROOT/'research_artifacts/r21_probability_lgb_officialcb_fivefold/fold0/result.json').read_text());assert old['parameters']==params('stats639')
files=['research_r91_features.py','research_r91_common.py','research_r91_worker.py','research_r91_driver.py','research_r91_prepare.py','research_r21_features.py','research_features.py','research_r23_loader.py','research_external_ablation.py','research_oof.py','research_r4_common.py','compare_validation.py','research_resume_policy_20261021.py','research_r82_result.json','data/train.csv']
for stage,K in [('fivefold',5),('threefold',3)]:
 z=refs(stage);assert len(set(z[0]['fold']))==K
 files.extend(f'research_artifacts/{n}_{stage}/oof.npz' for n in ['r21_probability_lgb_officialcb','r23_conditional_lean','r68_pure_ranknet'])
 for k in range(K):
  n=f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib';files.append(n);t=joblib.load(ROOT/n);assert t.columns+[FIELD]==tr.columns
for n in ['original_teacher_v1','r20_aux_officialcb','r21_conditional_officialcb']:files.extend([f'research_artifacts/{n}/result.json',f'research_artifacts/{n}/train_features.npz'])
p=dict(experiment='R91_one_max_observed_rating_surprisal_on_current_LGB84',frozen_utc=now(),hypothesis='A single extremelyinconsistentratingcanbedilutedbymean. Onefixedmaxcoordinateletsfinite63leafboostingexpressany-ratinganomalywithonecut; all13componentsandmeanarealreadyretained.',counterexample='[13,0,...,0]and[1,...,1]haveequalmean1butmax13vs1;no scores or labels usedtodesignoperator',evidence_limit='R82mean-surprisalproxyhadconsistentconflictenrichment,butitdoesnotprove max addsinformationorutility. DeterministicmaxaddsnoShannoninformation; onlytestsfinite-learnerrepresentation. No prior standalone maxsurprisaltrialfound.',new_feature=FIELD,source_columns=SURPRISALS,feature_columns=tr.columns,feature_width=85,learner_parameters=params('stats639'),unchanged='Fullcertified84prefixexact,source/auxmodelsunchanged;no maxthreshold/normalization/targetfitting;oldNNandRankNetfixed',fusion='Tnew=T_v12+.46875*(newLGB85-oldLGB84);nested.5tree/.5CE74then.875v11/.125Rank74logitpool,clip1e-6;weightsfixed',fixed_evaluation='All5beforequality;pairedDeLong95lower>0 ANDcorrectedresampledtalpha.025lower>0;onlyifpass3samegate;independentauditbeforefullLGBfit/testCSV',falsification='Eithergatefailsclose;no top-k/std/threshold/weight/seedrescueswithinR91',selection_warning='Historicallyreusedofficiallabels;3partitionnotuntouchedholdout;conditionalCIsomitadaptiveselection',resources=dict(device='CPU',threads=6,nice=10,host_RAM_reserve_GiB=4,own_worker_RSS_ceiling_GiB=8,disk_reserve_GiB=2,own_outputs_ceiling_MiB=256,preflight_disk_available_bytes=free,estimated_actual_all_stages_MiB='100-160',write_reservation_MiB=dict(perfold=32,OOF=96,fullfit_and_CSV=64),perfold_checkpoint_seconds=1200,estimated_fivefold_minutes='3-6',estimated_total_ifpasses_minutes='10-15',paid=False),preflight_entirefold0prefix84_exact=True,no_partial_quality=True,no_LB_tuning=True,parent_only_submitter=True,actual_top500_verified=False,frozen_sha256={n:sha(ROOT/n) for n in files})
dump(ROOT/'research_r91_plan.json',p);print(json.dumps(dict(plan_sha256=sha(ROOT/'research_r91_plan.json'),frozen_utc=p['frozen_utc'],input_files=len(files),width=85,free_MiB=free/2**20,total_output_cap_MiB=256,reserve_MiB=2048)))

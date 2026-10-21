import json,joblib,numpy as np
from research_r86_common import ROOT,sha,dump,now,refs
from research_oof import params
from research_resume_policy_20261021 import check_training
check_training();assert not (ROOT/'research_r86_plan.json').exists()
assert json.loads((ROOT/'research_r86_probe.json').read_text())['status']=='passed_synthetic_only'
cols=None
files=['research_r86_common.py','research_r86_worker.py','research_r86_driver.py','research_r86_prepare.py','research_r86_probe.py','research_r86_probe.json','research_r21_features.py','research_features.py','research_r23_loader.py','research_external_ablation.py','research_oof.py','research_r4_common.py','compare_validation.py','research_r41_plan.json','data/train.csv']
for stage,K in [('fivefold',5),('threefold',3)]:
 z=refs(stage);assert len(set(z[0]['fold']))==K
 files.extend(f'research_artifacts/{n}_{stage}/oof.npz' for n in ['r21_probability_lgb_officialcb','r23_conditional_lean','r68_pure_ranknet'])
 for k in range(K):
  path=f'research_artifacts/r21_probability_lgb_officialcb_{stage}/fold{k}/transformer.joblib';files.append(path);tr=joblib.load(ROOT/path)
  if cols is None:cols=tr.columns
  assert cols==tr.columns and len(cols)==84
for name in ['original_teacher_v1','r20_aux_officialcb','r21_conditional_officialcb']:files.extend([f'research_artifacts/{name}/result.json',f'research_artifacts/{name}/train_features.npz'])
vector=[1023 if c=='orig_teacher_lgb' else 255 for c in cols];assert vector.count(1023)==1 and vector[56]==1023
parameters=dict(params('stats639'),max_bin_by_feature=vector);a=np.load(ROOT/'research_artifacts/original_teacher_v1/train_features.npz');distinct=len(np.unique(a['values'][:,list(a['columns']).index('orig_teacher_lgb')]));assert distinct==692567
p=dict(experiment='R86_only_teacher_histogram_resolution255to1023_LGB84',frozen_utc=now(),hypothesis='Thedominantcontinuoussource teacher losesresolutionunder255binhistogram; allowing1023binsONLYonthisfeature maypreserverankinginformationwithsameleafcount/trees/otherfeatures',rationale='Teacherhas692567distinctcachedtrainvalues; currentlearnerhasmax_bin255andteacher93.975%-94.033%splitgain inR41audit. Thesearecapacityfacts,notproofofgeneralizationgain. No priorperfeaturebintrial found inrecordedplans.',assumption_limit='Finerbinscanoverfit/no guaranteedgain; thisisonefixednumericresolutiontest,nogrid. Distinctvaluesdonotprove255binsareinsufficient.',feature_columns=cols,feature_width=84,teacher_distinct_values=distinct,learner_parameters=parameters,only_changed_parameter='max_bin_by_feature[56]=1023,other83=255;globalmax_bin255;noR85monotoneconstraint',preflight_correction='Syntheticcategoricalbin includesunknownbin: firstassertionassumed3ratherthan4; beforefreeze/officialfit correctedchecktoassertnontargetbinsequal',fusion='Tnew=T_v12+.46875*(newLGB84-oldLGB84); nested.5tree/.5CE74logitpoolthen.875v11/.125Rank74,clip1e-6',fixed_evaluation='Finishall5beforequality;pairedDeLong95lower>0 ANDcorrectedresampledtalpha.025lower>0;onlyifpass3separatelyfittedpartitionssamegate;independentauditbeforefullfit/testCSV',falsification='Eithergatefailscloserecipe;nosmaller/largerbin/tree/weightrescue',no_partial_quality=True,no_LB_tuning=True,selection_warning='Historylabelsreusedextensively;3partitionisnotuntouchedholdout;conditionalCIsdonotcoverhistoricalselection',resources=dict(device='CPU',threads=6,nice=10,host_RAM_reserve_GiB=4,own_worker_RSS_ceiling_GiB=8,disk_reserve_GiB=2,own_outputs_ceiling_MiB=256,perfold_checkpoint_seconds=1200,estimated_fivefold_minutes='3-6',estimated_total_ifpasses_minutes='10-15',paid=False),authorization='Userbackgroundprojectcontinues;researchbranchonly,parentqualifiedsubmitunderactualquota;nonewcredentials',frozen_sha256={n:sha(ROOT/n) for n in files})
dump(ROOT/'research_r86_plan.json',p);print(json.dumps(dict(plan_sha256=sha(ROOT/'research_r86_plan.json'),frozen_utc=p['frozen_utc'],frozen_files=len(files))))

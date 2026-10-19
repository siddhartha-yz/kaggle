"""Diagnostic-only existing OOF error review; no refits, weight search or rescue."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.special import logit
ROOT=Path(__file__).resolve().parent
paths={72:'r72_largecontext_raw21',73:'r73_joint_density',74:'r74_member_origins',75:'r75_tabdpt_raw21',76:'r76_mitra_v2',77:'r77_ebm_raw21'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 base=np.load(ROOT/'research_artifacts/r68_pure_ranknet_fivefold/oof.npz');y=base['target'];p=base['candidate'];l=logit(np.clip(p,1e-6,1-1e-6));rng=np.random.default_rng(202610191224);pos=rng.choice(np.flatnonzero(y==1),1000000);neg=rng.choice(np.flatnonzero(y==0),1000000);correct=p[pos]>p[neg];wrong=p[pos]<p[neg];entries=[];res=[]
 for k,name in paths.items():
  path=ROOT/f'research_artifacts/{name}_fivefold/oof.npz';z=np.load(path);assert all(np.array_equal(z[q],base[q]) for q in ['id','target','fold']) and np.array_equal(z['reference_v12'],p)
  q=z['pred'];c=z['candidate'];a=logit(np.clip(q,1e-6,1-1e-6));res.append(y-q)
  entries.append(dict(route=k,oof_sha256=sha(path),OOF_logit_correlation_with_v12=float(np.corrcoef(l,a)[0,1]),probability_residual_correlation_with_v12=float(np.corrcoef(y-p,y-q)[0,1]),native_repairs_of_sampled_v12_wrong_pairs=int(np.sum(wrong&(q[pos]>q[neg]))),native_ruins_of_sampled_v12_right_pairs=int(np.sum(correct&(q[pos]<q[neg]))),frozen_candidate_repairs=int(np.sum(wrong&(c[pos]>c[neg]))),frozen_candidate_ruins=int(np.sum(correct&(c[pos]<c[neg]))),native_pair_ties=int(np.sum(q[pos]==q[neg])),no_weight_or_recipe_rescue=True))
 out=dict(status='diagnostic_existing_predictions_only',checked_utc=datetime.now(timezone.utc).isoformat(),reference='Actualformalv12',pair_sample=dict(seed=202610191224,count=1000000,positive_negative_independent_with_replacement=True,baseline_wrong=int(wrong.sum()),baseline_right=int(correct.sum()),baseline_tied=int((~(wrong|correct)).sum())),routes=entries,native_residual_correlations=dict(route_order=list(paths),matrix=np.corrcoef(res).tolist()),interpretation='Correlatederrorsandnativepairrepairsaloneareinsufficient:allfixedrecentcandidatesfailedoriginalgates;repair/ruinratesareconditionedonverydifferentbasepaircounts. Do notreusefailuresasnewcandidates orsearchweights. R74isexistingmodeloriginrepooling,notnewtrainedfamily.',next_hypothesis='TabSTAR addsreal-world400dataset semanticfeature/targetpretrainingandLoRAadaptationonallownoutertrainingrows, unlike R75no-updateICL/R72contextICL or R76small-episodeadaptation. It canlearntask-specificsemanticembeddings/ratinginteractions ratherthanonlychangecontext/bagcount. Raw21containslessengineeredinformationthanv12NN74;complementarityandgainarenotestablisheduntilprospectiveOOF.',prospective_fusion='ONE small1/16TabSTAR shareagainst15/16actualv12inlogitspace;chosenbeforeanyTabSTARofficialfit/score,notfittofailedOOFs. No extraweights orrescuegrid.',new_classifier_fits=0,new_candidate_quality_tests=0,official_SAT_reused_for_diagnostics=True,untouched_holdout=False,actual_top500_verified=False)
 (ROOT/'research_recent_complementarity_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(sample=out['pair_sample'],routes=entries),indent=2))
if __name__=='__main__':main()

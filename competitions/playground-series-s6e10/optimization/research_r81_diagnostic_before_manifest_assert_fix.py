"""Frozen existing-prediction diagnostic only. No classifier or prediction calls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,resource,gc
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
NAMES=['confident_agree','confident_conflict','uncertain']
NEAR=['Flight Distance','Age','Gender','Class','Customer Type','Type of Travel']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def matrix(y,score,bucket,names=NAMES):
 m=int(y.sum());n=len(y)-m;den=m*n;out=[];total=0.;pairmass=np.zeros(len(names));lossmass=np.zeros(len(names))
 for i,a in enumerate(names):
  p=score[(y==1)&(bucket==i)]
  for j,b in enumerate(names):
   q=np.sort(score[(y==0)&(bucket==j)]);pairs=len(p)*len(q)
   if pairs:
    lo=np.searchsorted(q,p,'left');hi=np.searchsorted(q,p,'right');ties=int(np.sum(hi-lo));wrong=int(np.sum(len(q)-hi));loss=wrong+.5*ties
   else:ties=wrong=0;loss=0.
   total+=loss;pairmass[i]+=.5*pairs/den;pairmass[j]+=.5*pairs/den;lossmass[i]+=.5*loss/den;lossmass[j]+=.5*loss/den
   out.append(dict(positive_bucket=a,negative_bucket=b,pairs=pairs,discordant_pairs=wrong,ties=ties,loss_pair_equivalents=loss,cell_error_rate=loss/pairs if pairs else None,global_AUC_loss_contribution=loss/den))
 for a in out:a['fraction_of_all_loss']=a['loss_pair_equivalents']/total if total else 0.
 return dict(AUC=1-total/den,AUC_loss=total/den,positive_rows=m,negative_rows=n,total_pairs=den,loss_pair_equivalents=total,cells=out,half_endpoint_attribution=[dict(bucket=name,pair_mass_fraction=float(pairmass[i]),global_AUC_loss_contribution=float(lossmass[i]),fraction_of_all_loss=float(lossmass[i]/(total/den))) for i,name in enumerate(names)])
def endpoint_loss(y,p):
 m=int(y.sum());n=len(y)-m;den=m*n;a=np.sort(p[y==1]);b=np.sort(p[y==0]);v=np.empty(len(y));pos=p[y==1];neg=p[y==0];lo=np.searchsorted(b,pos,'left');hi=np.searchsorted(b,pos,'right');v[y==1]=(n-hi+.5*(hi-lo))/(2*den);lo=np.searchsorted(a,neg,'left');hi=np.searchsorted(a,neg,'right');v[y==0]=(lo+.5*(hi-lo))/(2*den);return v
def key(frame):
 x=[]
 for c in NEAR:
  a=frame[c].astype('float64').astype('string') if pd.api.types.is_numeric_dtype(frame[c]) else frame[c].astype('string');x.append(a.fillna('__MISSING__'))
 out=x[0]
 for a in x[1:]:out=out+'\x1f'+a
 return out
def main():
 tick=time.monotonic();plan=json.loads((ROOT/'research_r81_plan.json').read_text());ph=sha(ROOT/'research_r81_plan.json');assert sha(__file__)==plan['diagnostic_source_sha256']
 for n,h in plan['input_sha256'].items():assert sha(ROOT/n)==h,n
 assert not (ROOT/'research_r81_result.json').exists(),'Do not repeat diagnostic to select thresholds'
 frame=pd.read_csv(ROOT/'data/train.csv');ids=frame.id.to_numpy();y=frame.satisfaction.to_numpy('int8');raw=frame.drop(columns=['id','satisfaction']);best=np.load(ROOT/'research_artifacts/r68_pure_ranknet_fivefold/oof.npz');old=np.load(ROOT/'research_artifacts/r23_conditional_lean_fivefold/oof.npz');tree=np.load(ROOT/'research_artifacts/r21_probability_lgb_officialcb_fivefold/oof.npz');teacher=np.load(ROOT/'research_artifacts/original_teacher_v1/train_features.npz')
 for z in [best,old,tree]:assert np.array_equal(z['id'],ids) and np.array_equal(z['target'],y) and np.array_equal(z['fold'],best['fold'])
 assert np.array_equal(teacher['id'],ids);q=teacher['values'][:,list(teacher['columns']).index('orig_teacher_lgb')];p=best['candidate'];fold=best['fold'];assert np.array_equal(old['candidate'],best['reference_v11']);assert all(np.isfinite(a).all() and ((a>=0)&(a<=1)).all() for a in [q,p]);conf=(q<=.1)|(q>=.9);agree=(q>=.5)==y;bucket=np.where(~conf,2,np.where(agree,0,1)).astype('int8');assert set(np.unique(bucket))=={0,1,2}
 scores={'v12':p,'v11':old['candidate'],'tree_component':tree['tree'],'CE_NN74_component':old['pred'],'RankNet74_component':best['pred'],'original_teacher':q};global_results={n:matrix(y,s,bucket) for n,s in scores.items()};expected=json.loads((ROOT/'research_r68_result.json').read_text())['validation']['fivefold']['pooled']['candidate_auc'];assert abs(global_results['v12']['AUC']-expected)<1e-12;fold_results={str(k):matrix(y[fold==k],p[fold==k],bucket[fold==k]) for k in range(5)};ep=endpoint_loss(y,p);assert abs(ep.sum()-global_results['v12']['AUC_loss'])<1e-12
 rows=[]
 for i,name in enumerate(NAMES):
  m=bucket==i;rows.append(dict(bucket=name,rows=int(m.sum()),positive_rows=int(y[m].sum()),positive_rate=float(y[m].mean()),teacher_probability_quantiles=np.quantile(q[m],[0,.1,.5,.9,1]).tolist(),v12_mean_probability=float(p[m].mean()),global_AUC_loss_endpoint_contribution=float(ep[m].sum())))
 print('EXACT_PAIR_MATRICES_COMPLETE',flush=True)
 source=pd.read_csv(ROOT/'research_sources/original_cc0/data.csv',usecols=list(raw));keep=np.load(ROOT/'research_sources/original_safe_mask.npy');assert len(source)==129880 and keep.sum()==129859;source=source.loc[keep].reset_index(drop=True);sk=key(source);qk=key(raw);sm=pd.DataFrame({'key':sk,'source_row':np.arange(len(source))});remaining=[c for c in raw if c not in NEAR];assert len(remaining)==15 and all(pd.api.types.is_numeric_dtype(raw[c]) for c in remaining);sv=source[remaining].to_numpy('float64');qv=raw[remaining].to_numpy('float64');seen=np.zeros((5,len(source)),bool);tightseen=np.zeros_like(seen);matched=np.zeros(len(raw),bool);tightmatched=np.zeros_like(matched);pairs=np.zeros(5,'int64');tightpairs=np.zeros(5,'int64');exactpairs=np.zeros(5,'int64');exactrows=np.zeros_like(matched)
 for start in range(0,len(raw),50000):
  end=min(len(raw),start+50000);j=pd.DataFrame({'key':qk.iloc[start:end].to_numpy(),'query_row':np.arange(start,end)}).merge(sm,on='key',how='inner',sort=False);a=j.query_row.to_numpy();b=j.source_row.to_numpy();matched[a]=True;seen[fold[a],b]=True;pairs+=np.bincount(fold[a],minlength=5)
  for lo in range(0,len(a),100000):
   aa=a[lo:lo+100000];bb=b[lo:lo+100000];x=qv[aa];z=sv[bb];equal=((x==z)|(np.isnan(x)&np.isnan(z))).sum(axis=1);ok=equal>=14;ex=equal==15;tightmatched[aa[ok]]=True;tightseen[fold[aa[ok]],bb[ok]]=True;tightpairs+=np.bincount(fold[aa[ok]],minlength=5);exactpairs+=np.bincount(fold[aa[ex]],minlength=5);exactrows[aa[ex]]=True
  del j;gc.collect()
  print('SOURCE_NEAR_SCAN',end,len(raw),flush=True)
 assert matched.sum()==json.loads((ROOT/'original_overlap_audit.json').read_text())['synthetic_train_rows_near_key_match_original'];assert exactrows.sum()==exactpairs.sum()==0
 near=[]
 for k in range(5):
  m=fold==k;other=np.arange(5)!=k;near.append(dict(fold=k,validation_rows=int(m.sum()),source_pool_rows=len(source),near6_validation_rows=int(matched[m].sum()),near6_validation_fraction=float(matched[m].mean()),near6_candidate_pairs=int(pairs[k]),near6_distinct_validation_keys=int(qk[m&matched].nunique()),source_rows_near6_any_validation_row=int(seen[k].sum()),source_rows_near6_both_validation_and_outertrain=int((seen[k]&seen[other].any(axis=0)).sum()),tight20_within_near6_validation_rows=int(tightmatched[m].sum()),tight20_candidate_pairs=int(tightpairs[k]),source_rows_tight20_any_validation_row=int(tightseen[k].sum()),source_rows_tight20_both_validation_and_outertrain=int((tightseen[k]&tightseen[other].any(axis=0)).sum()),remaining_full21_exact_pairs=int(exactpairs[k])))
 cross=[]
 for i,name in enumerate(NAMES):
  for label,mask in [('near6',matched),('tight20_within_near6',tightmatched)]:
   for flag in [False,True]:
    m=(bucket==i)&(mask==flag);cross.append(dict(bucket=name,standard=label,matched=flag,rows=int(m.sum()),global_AUC_loss_endpoint_contribution=float(ep[m].sum()),fraction_of_v12_all_loss=float(ep[m].sum()/ep.sum())))
 gain=[]
 for a,b in zip(global_results['v12']['cells'],global_results['v11']['cells']):gain.append(dict(positive_bucket=a['positive_bucket'],negative_bucket=a['negative_bucket'],AUC_gain_v12_minus_v11=b['global_AUC_loss_contribution']-a['global_AUC_loss_contribution']))
 result=dict(status='completed_descriptive_existing_predictions_only',completed_utc=now(),plan_sha256=ph,bucket_definitions=plan['buckets'],buckets=rows,ranking_matrices=global_results,fold_v12_ranking_matrices=fold_results,v12_minus_v11_gain_decomposition=gain,source_near_definitions=plan['source_near'],source_near_by_outer_fold=near,source_near_by_label_defined_bucket=cross,source_summary=dict(safe_rows=len(source),near6_official_train_rows=int(matched.sum()),near6_source_pairs=int(pairs.sum()),source_rows_near6_any_official_train_row=int(seen.any(axis=0).sum()),tight20_within_near6_official_train_rows=int(tightmatched.sum()),tight20_source_pairs=int(tightpairs.sum()),remaining_full21_exact_pairs=int(exactpairs.sum())),new_classifier_fits=0,new_model_inference=0,new_candidates_evaluated=0,new_weights_or_thresholds_searched=0,official_labels_used_for_description=True,labels_changed=False,raw_row_predictions_saved=False,seconds=time.monotonic()-tick,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,limitations=plan['limitations']);dump(ROOT/'research_r81_result.json',result);print('COMPLETE',result['seconds'],result['source_summary'],flush=True)
if __name__=='__main__':main()

"""Read-only exploratory v6 OOF error/ensemble diagnostics; no new model fitting."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='6'
os.environ['MPLCONFIGDIR']='/tmp/kaggle-s6e10-mpl'
import numpy as np,pandas as pd,torch,joblib,cloudpickle
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def concordance(y,p):
 pos=np.sort(p[y==1]);neg=np.sort(p[y==0]);c=np.empty(len(y));q=p[y==1];c[y==1]=(np.searchsorted(neg,q,'left')+np.searchsorted(neg,q,'right'))/(2*len(neg));q=p[y==0];c[y==0]=1-(np.searchsorted(pos,q,'left')+np.searchsorted(pos,q,'right'))/(2*len(pos));assert abs(c[y==1].mean()-roc_auc_score(y,p))<1e-12;return c

def main():
 torch.set_num_threads(6);torch.set_num_interop_threads(1);start=time.monotonic();out=ROOT/'research_artifacts/w3_diagnostics_v1';out.mkdir(exist_ok=True);df=pd.read_csv(ROOT/'data/train.csv');raw=df.drop(columns=['id','satisfaction']);y=df.satisfaction.astype('int8').to_numpy();nn=np.load(ROOT/'research_artifacts/gpu_value_raw_oof_n2/oof.npz');fold=nn['fold'];v5=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');assert all(np.array_equal(nn[k],v5[k]) for k in ['id','target','fold']);assert np.array_equal(nn['id'],df.id) and np.array_equal(nn['target'],y);v6=nn['candidate'];assert np.array_equal(v6,.75*v5['pred']+.25*nn['pred']);preds={'v6':v6,'v5':v5['pred'],'realmlp_mean':nn['pred']}
 for name,folder in [('lgb_original','lgb_stats_orig_both_oof'),('cb8_original','cb_stats_orig_both_oof'),('residual_prior','residual_prior05_oof_t581')]:
  z=np.load(ROOT/'research_artifacts'/folder/'oof.npz');assert all(np.array_equal(z[k],nn[k]) for k in ['id','target','fold']);preds[name]=z['pred']
 z=np.load(ROOT/'research_artifacts/cb_deep_oof/raw_oof.npz');assert all(np.array_equal(z[k],nn[k]) for k in ['id','target','fold']);preds['cb10']=z['pred']
 members=np.full((2,len(y)),np.nan,dtype=np.float32);freq=np.zeros(len(y));differences=[]
 with threadpool_limits(6):
  for k in range(5):
   fi=np.flatnonzero(fold!=k);vi=np.flatnonzero(fold==k);folder=ROOT/'research_artifacts/gpu_value_raw_oof_n2'/f'fold{k}';r=json.loads((folder/'result.json').read_text());assert sha(folder/'model.pkl')==r['model_sha256'];tr=joblib.load(folder/'transformer.joblib');X=tr.transform(raw.iloc[vi].reset_index(drop=True),pd.DataFrame(index=np.arange(len(vi))))
   with (folder/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
   module=model.cv_alg_interface_.model;assert model.cv_alg_interface_.device=='cpu';ds=model.x_converter_.transform(X);module.model.eval();parts=[]
   with torch.no_grad():
    for batch in module.get_predict_dataloader(ds):
     logits=module.model(batch)['x_cont'];assert logits.shape[0]==2 and logits.shape[-1]==2;parts.append(torch.softmax(logits,dim=-1)[:,:,1].cpu())
   p=torch.cat(parts,dim=1).numpy();delta=float(np.max(np.abs(p.mean(axis=0)-nn['pred'][vi])));assert delta<1e-5;members[:,vi]=p;differences.append(delta);counts=raw.iloc[fi]['Flight Distance'].value_counts();freq[vi]=raw.iloc[vi]['Flight Distance'].map(counts).fillna(0);del model,module,ds,X,parts;gc.collect();print('MEMBERS',k,delta,flush=True)
 preds['realmlp_member0']=members[0];preds['realmlp_member1']=members[1];np.savez_compressed(out/'member_oof.npz',id=df.id.to_numpy(),target=y,fold=fold,member0=members[0],member1=members[1])
 metrics={name:{'auc':float(roc_auc_score(y,p)),'logloss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p))} for name,p in preds.items()};corr=np.corrcoef(np.stack([y-p for p in preds.values()]));ext=np.load(ROOT/'research_artifacts/original_teacher_v1/train_features.npz');assert np.array_equal(ext['id'],df.id);extdf=pd.DataFrame(ext['values'],columns=ext['columns']);c5=concordance(y,preds['v5']);c6=concordance(y,v6);npos=int(y.sum());nneg=len(y)-npos
 groups={c:raw[c].astype(str) for c in ['Class','Customer Type','Type of Travel','Gender']};groups['age_bin']=pd.cut(raw.Age,[-np.inf,25,40,60,np.inf],labels=['<=25','26-40','41-60','61+']).astype(str);groups['distance_train_count']=pd.cut(freq,[-1,0,9,49,199,np.inf],labels=['unseen','1-9','10-49','50-199','200+']).astype(str);groups['original_near_match']=np.where(extdf.orig_lookup1_logcount>0,'match','no_match');dis=np.abs(members[0]-members[1]);groups['neural_disagreement']=pd.qcut(dis,4,labels=['q1','q2','q3','q4'],duplicates='drop').astype(str)
 slices={}
 for feature,values in groups.items():
  result={}
  for value in sorted(np.unique(values)):
   mask=np.asarray(values==value);yp=y[mask];pp=v6[mask];positive=mask&(y==1);negative=mask&(y==0);gap=(float((1-c6[positive]).sum())/(2*npos)+float((1-c6[negative]).sum())/(2*nneg));gain=(float((c6[positive]-c5[positive]).sum())/(2*npos)+float((c6[negative]-c5[negative]).sum())/(2*nneg));result[str(value)]={'rows':int(mask.sum()),'positive_rate':float(yp.mean()),'auc_v6':float(roc_auc_score(yp,pp)) if len(np.unique(yp))==2 else None,'auc_v5':float(roc_auc_score(yp,preds['v5'][mask])) if len(np.unique(yp))==2 else None,'logloss_v6':float(log_loss(yp,pp,labels=[0,1])),'contribution_to_global_auc_gap':gap,'contribution_to_v6_minus_v5_auc':gain}
  slices[feature]=result
 highwrong=((y==0)&(v6>.9))|((y==1)&(v6<.1));teacher=extdf.orig_teacher_lgb.to_numpy();teacherwrong=((y==0)&(teacher>.5))|((y==1)&(teacher<.5));report={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'baseline_public_record':{'auc':.96058,'rank_snapshot':858,'total_ranked':2208,'observed_utc':'2026-10-18T04:00:00+00:00'},'metrics':metrics,'residual_correlation_names':list(preds),'residual_correlations':corr.tolist(),'member_mean_vs_saved_max_abs_diff_by_fold':differences,'two_member_average_gain_over_member_aucs':float(metrics['realmlp_mean']['auc']-.5*(metrics['realmlp_member0']['auc']+metrics['realmlp_member1']['auc'])),'high_confidence_wrong':{'rows':int(highwrong.sum()),'fraction':float(highwrong.mean()),'original_lgb_teacher_also_wrong_fraction':float(teacherwrong[highwrong].mean()),'mean_nn_member_disagreement':float(dis[highwrong].mean())},'slices':slices,'prediction_files_from_own_audited_models_only':True,'new_fits_performed':False,'limitation':'Exploratory diagnostics use reused OOF labels to choose hypotheses; future conditional intervals do not remove that model-selection bias.'};dump(ROOT/'research_w3_error_analysis.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['slices','residual_correlations']},indent=2),flush=True)
if __name__=='__main__':main()

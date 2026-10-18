"""One nativeRealMLP16dim representation contrast, honest cached CPU5inner control."""
from research_w3_common import *
import research_w3_nested_neural as fitlib
import argparse,time,gc,psutil,resource
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();start_guard();torch.set_num_threads(6);torch.set_num_interop_threads(1);frozen=json.loads((ROOT/'research_w3_realmlp_width_plan.json').read_text());config=frozen['config'];fitlib.pp=lambda:{'nn_config':config}
 def memory_guard():
  rss=psutil.Process().memory_info().rss;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;available=psutil.virtual_memory().available;assert peak<=5*1024**3 and available>=3*1024**3,'Width16RAMguard';return {'rss_bytes':rss,'peak_rss_bytes':peak,'available_ram_bytes':available}
 fitlib.guard=memory_guard
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_realmlp_width_{prev}/result.json').read_text())['eligible_for_next']
 assert fitlib.hashfile(ROOT/'data/train.csv')=='2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36';df,raw,y,splits,ref,old,fold=load_split(a.split);out=ROOT/f'research_artifacts/w3_realmlp_width_{a.split}';out.mkdir(exist_ok=True);allpred=np.full(len(y),np.nan);comps=[]
 with threadpool_limits(6):
  for k,(fi,vi) in enumerate(splits):
   run=out/f'fold{k}';run.mkdir(exist_ok=True);rec={'status':'training','started_utc':now(),'plan_sha256':sha(ROOT/'research_w3_realmlp_width_plan.json'),'runner_sha256':sha(__file__),'utility_sha256':sha(fitlib.__file__),'train_rows':len(fi),'valid_rows':len(vi),'config':config};dump(run/'result.json',rec)
   if a.split=='primary':
    it,im=train_test_split(fi,test_size=.2,stratify=y[fi],random_state=20261020);baseline=ROOT/'research_artifacts/w3_nested_neural_primary/fold0/nn_inner';b=json.loads((baseline/'result.json').read_text());assert b['status']=='complete' and b['config']['embedding_size']==5 and all(v==b['config'][key] for key,v in config.items() if key!='embedding_size') and b['model_sha256']==fitlib.hashfile(baseline/'model.pkl') and b['prediction_sha256']==fitlib.hashfile(baseline/'prediction.npz');z=np.load(baseline/'prediction.npz');assert np.array_equal(z['idx'],im) and np.array_equal(z['id'],df.id.iloc[im]) and np.array_equal(z['target'],y[im]);p,nr=fitlib.neural(raw,y,df,it,im,run/'inner16');assert nr['training_indices_sha256']==b['training_indices_sha256'] and nr['training_target_sha256']==b['training_target_sha256'];c=compare(y[im],z['pred'],p);screen={'status':'complete','baseline_config':b['config'],'new_config':config,'inner_vs_CPU5':c,'eligible_for_outer':c['delta_auc']>0,'old_control_retrained':False,'completed_utc':now()};dump(run/'inner_comparison.json',screen)
    if c['delta_auc']<=0:rec.update(status='complete',outcome='training_only_inner_capacity_test_rejected',inner_comparison=screen,completed_utc=now());dump(run/'result.json',rec);dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'Innerwidth16didnotbeatcachedCPU5','outer_validation_evaluated':False,'inner_comparison':screen,'completed_utc':now()});print(json.dumps(screen,indent=2),flush=True);return
   start_guard();pred,nr=fitlib.neural(raw,y,df,fi,vi,run/'outer16');r=ref[vi] if a.split=='oof' else ref;o=old[vi] if a.split=='oof' else old;candidate=r+.25*(pred-o);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();c=compare(y[vi],r,candidate);allpred[vi]=pred;comps.append(c);np.savez_compressed(run/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=r,old_nn=o);rec.update(status='complete',vs_v6=c,model=nr,completed_utc=now());dump(run/'result.json',rec);print(json.dumps(rec,indent=2),flush=True);gc.collect()
   if a.split=='oof' and c['delta_auc']<=0:dump(out/'result.json',{'status':'complete','eligible_for_next':False,'reason':'nonpositive fold','fold_comparisons':comps,'completed_utc':now()});return
 vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];pred=allpred[vi];candidate=ref+.25*(pred-old);c=compare(y[vi],ref,candidate);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=pred,candidate=candidate,v6=ref,old_nn=old,**({'fold':fold} if fold is not None else {}));res={'status':'complete','split':a.split,'outer_validation_evaluated':True,'vs_v6':c,'fold_comparisons':comps,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in comps),'completed_utc':now()};dump(out/'result.json',res);print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'research_w3_realmlp_width_last_failure.json',{'status':'failed','ended_utc':now(),'error':repr(e)});raise

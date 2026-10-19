"""One frozen equal average within and between supervised neural families."""
from research_w3_common import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args();pp=json.loads((ROOT/'research_w3_family_fusion_plan.json').read_text());df,_,y,splits,ref,old,fold=load_split(a.split);vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];families={};sources={}
 for family,runs in pp['families'].items():
  pieces=[]
  for primary_run in runs:
   run=primary_run.replace('_primary','_'+a.split);p=ROOT/f'research_artifacts/{run}/validation.npz';z=np.load(p);assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);pred=z['pred'];assert len(pred)==len(vi) and np.isfinite(pred).all();pieces.append(pred);sources[run]=sha(p)
  families[family]=np.mean(pieces,axis=0)
 means=np.column_stack(list(families.values()));candidate=.5*ref+.5*means.mean(axis=1);control=.5*ref+.5*old;c=compare(y[vi],ref,candidate);matched=compare(y[vi],control,candidate);byfold=[compare(y[vi][fold==k],ref[fold==k],candidate[fold==k]) for k in range(5)] if fold is not None else [];eligible=gate(c,a.split) and matched['delta_auc']>0 and all(x['delta_auc']>0 for x in byfold);out=ROOT/f'research_artifacts/w3_family_fusion_{a.split}';out.mkdir(exist_ok=True);assert not (out/'result.json').exists(),'Do not overwrite a completed ensemble decision';np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],candidate=candidate,v6=ref,control=control,**families,**({'fold':fold} if fold is not None else {}));rec={'status':'complete','completed_utc':now(),'split':a.split,'plan_sha256':sha(ROOT/'research_w3_family_fusion_plan.json'),'source_prediction_sha256':sources,'aggregation_has_no_new_fits':True,'vs_v6':c,'vs_matched_control':matched,'fold_comparisons':byfold,'family_deviation_correlation_without_targets':np.corrcoef(means-ref[:,None],rowvar=False).tolist(),'families':list(families),'eligible_for_next':eligible};dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()

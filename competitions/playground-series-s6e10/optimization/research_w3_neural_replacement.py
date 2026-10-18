"""One fixed4-family average inside the existing25% neural component."""
from research_w3_common import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['primary','confirm','oof'],required=True);a=ap.parse_args()
 for prev in (['primary'] if a.split=='confirm' else ['primary','confirm'] if a.split=='oof' else []):assert json.loads((ROOT/f'research_artifacts/w3_neural_replacement_{prev}/result.json').read_text())['eligible_for_next']
 familyplan=json.loads((ROOT/'research_w3_family_fusion_plan.json').read_text());df,_,y,splits,ref,old,fold=load_split(a.split);vi=np.arange(len(y)) if a.split=='oof' else splits[0][1];families=[old];sources={}
 for family,runs in familyplan['families'].items():
  predictions=[]
  for name in runs:
   path=ROOT/f'research_artifacts/{name.replace("_primary","_"+a.split)}/validation.npz';z=np.load(path);assert np.array_equal(z['idx'],vi) and np.array_equal(z['id'],df.id.iloc[vi]) and np.array_equal(z['target'],y[vi]);pred=z['pred'].astype('float64');assert np.isfinite(pred).all();predictions.append(pred);sources[str(path.relative_to(ROOT))]=sha(path)
  families.append(np.mean(predictions,axis=0))
 assert len(families)==4;neural=np.mean(families,axis=0);candidate=ref+.25*(neural-old);assert np.isfinite(candidate).all() and ((candidate>=0)&(candidate<=1)).all();c=compare(y[vi],ref,candidate);byfold=[compare(y[vi][fold==k],ref[fold==k],candidate[fold==k]) for k in range(5)] if fold is not None else [];out=ROOT/f'research_artifacts/w3_neural_replacement_{a.split}';out.mkdir(exist_ok=False);np.savez_compressed(out/'validation.npz',idx=vi,id=df.id.iloc[vi].to_numpy(),target=y[vi],pred=neural,candidate=candidate,v6=ref,old_nn=old,**({'fold':fold} if fold is not None else {}));rec={'status':'complete','completed_utc':now(),'split':a.split,'plan_sha256':sha(ROOT/'research_w3_neural_replacement_plan.json'),'source_sha256':sources,'new_fits':0,'tree_mass':.75,'neural_mass':.25,'vs_v6':c,'fold_comparisons':byfold,'eligible_for_next':gate(c,a.split) and all(x['delta_auc']>0 for x in byfold)};dump(out/'result.json',rec);print(json.dumps(rec,indent=2),flush=True)
if __name__=='__main__':main()

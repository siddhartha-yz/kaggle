"""Read-only v5 verification; no fitting, credentials, or submission."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from compare_validation import compare
ROOT=Path(__file__).resolve().parent
expected={'train.csv':'2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36','test.csv':'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6','sample_submission.csv':'6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71'}
for n,h in expected.items():assert hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()==h
csv=ROOT/'submission_cb_deep_v5.csv';csvhash=hashlib.sha256(csv.read_bytes()).hexdigest();assert csvhash=='0f9a978051e4f0fa164928f3b975a71346b60477959ee5cf50abb580a706e872'
df=pd.read_csv(csv);sample=pd.read_csv(ROOT/'data/sample_submission.csv');z=np.load(ROOT/'research_artifacts/cb_deep_full_v1/test_components.npz');assert list(df)==['id','satisfaction'] and len(df)==299844 and df.id.is_unique and df.id.equals(sample.id) and np.array_equal(z['id'],df.id)
weights={'lgb_original':.46875,'cb_original':.09375,'residual_prior05':.1875,'cb_deep':.25};pred=sum(w*z[k] for k,w in weights.items());assert np.isfinite(df.satisfaction).all() and df.satisfaction.between(0,1).all();delta=float(np.max(np.abs(pred-df.satisfaction)));assert delta<1e-14
train=pd.read_csv(ROOT/'data/train.csv',usecols=['id','satisfaction']);o=np.load(ROOT/'research_artifacts/cb_deep_oof/candidate_oof.npz');ref=np.load(ROOT/'research_artifacts/lgb_stats_orig_both_oof/oof.npz');assert all(np.array_equal(o[k],ref[k]) for k in ['id','target','fold']);assert np.array_equal(o['id'],train.id) and np.array_equal(o['target'],train.satisfaction.astype('int8'));comparison=compare(o['target'],ref['pred'],o['pred'])
print(json.dumps({'status':'verified','csv_sha256':csvhash,'rows':len(df),'weights':weights,'csv_component_max_abs_diff':delta,'oof_vs_v4':comparison,'training_started':False},indent=2))

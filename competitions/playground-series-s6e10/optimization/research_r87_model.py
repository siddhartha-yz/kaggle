"""One matched CE neural representation; tree/RankNet components are unchanged."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='6'
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parent
import json,hashlib
import numpy as np,torch
from scipy.special import expit,logit
from research_r8_probe import BoundedRealMLP
from research_r8_streaming import StreamingTabNNModule,streaming_backend
from research_r87_features import OfficialTeacherNeuralFeatures
from research_r87_loader import auxiliary
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(epochs=3):
 c=dict(json.loads((ROOT/'research_r23_plan.json').read_text())['config']);c.update(n_epochs=epochs,stop_epoch=epochs);assert c['n_ens']==32 and c['batch_size']==256 and c['n_threads']==6;return c
def candidate(tree,neural,rank):
 v11=expit(.5*logit(np.clip(tree,1e-6,1-1e-6))+.5*logit(np.clip(neural,1e-6,1-1e-6)))
 return expit(.875*logit(np.clip(v11,1e-6,1-1e-6))+.125*logit(np.clip(rank,1e-6,1-1e-6)))
def references(stage):
 tree=np.load(ROOT/f'research_artifacts/r21_probability_lgb_officialcb_{stage}/oof.npz');nn=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{stage}/oof.npz');best=np.load(ROOT/f'research_artifacts/r68_pure_ranknet_{stage}/oof.npz')
 for a in [tree,nn]:assert all(np.array_equal(a[n],best[n]) for n in ['id','target','fold'])
 assert np.max(np.abs(candidate(tree['tree'],nn['pred'],best['pred'])-best['candidate']))<1e-12
 return best,tree['tree']

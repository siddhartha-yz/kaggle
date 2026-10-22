"""Synthetic-only equivalence, including duplicates, empty keys and string sentinels."""
import json,numpy as np,pandas as pd,time
from research_r93_common import ROOT,sha,dump,reserve,now
from research_r93_repair_helpers import string_support

def original(a,b,sample):
 sa=a[sample]
 return dict(full_train_unique=len(np.unique(a)),full_test_unique=len(np.unique(b)),full_test_rows_unseen_train=int((~np.isin(b,np.unique(a))).sum()),equal_size=len(b),equal_train_rows_unseen_test=int((~np.isin(sa,np.unique(b))).sum()),equal_test_rows_unseen_train=int((~np.isin(b,np.unique(sa))).sum()))
rng=np.random.default_rng(20261022);values=np.array(['0.0','-0.0','1','1.0','nan','__MISSING__','a\x1fb','a\x1fc','é','中文','A','a'],dtype=object);tick=time.monotonic()
for k in range(120):
 n=int(rng.integers(0,240));m=int(rng.integers(0,120));a=rng.choice(values,n);b=rng.choice(values,m);sample=rng.choice(n,size=min(n,m),replace=False)
 assert original(a,b,sample)==string_support(a,b,sample)
# Representative multi-field canonical string keys, including near equal numeric text.
for width in [1,3,4,6]:
 a=np.array(['\x1f'.join(rng.choice(values,width)) for _ in range(180)],dtype=object);b=np.array(['\x1f'.join(rng.choice(values,width)) for _ in range(90)],dtype=object);sample=rng.choice(len(a),size=len(b),replace=False);assert original(a,b,sample)==string_support(a,b,sample)
record=dict(status='passed_synthetic_only',completed_utc=now(),source_sha256=sha(__file__),optimized_helper_sha256=sha(ROOT/'research_r93_repair_helpers.py'),cases=124,equivalence='Exactall6supportfieldsforcanonicalstrings;uniqueorderingnotusedinthemetrics',no_official_data_loaded=True,no_models_fitted=True,no_thresholds_or_splits_changed=True,seconds=time.monotonic()-tick)
reserve();dump(ROOT/'research_r93_membership_equivalence.json',record);print(json.dumps(record))

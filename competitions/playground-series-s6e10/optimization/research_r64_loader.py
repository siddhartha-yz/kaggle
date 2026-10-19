"""Only fixed aggregate threefold qualification; no individual failed-route gate override."""
from research_r23_loader import ROOT,STAGE,pool,load_split as baseline
import json,sys,importlib,numpy as np
VIEW=sys.argv[sys.argv.index('--view')+1]
P=json.loads((ROOT/'research_r64_plan.json').read_text());MEMBER=next(r for r in P['members'] if r['name']==VIEW)
assert VIEW!='r23'
auxiliary=importlib.import_module(MEMBER['loader_module']).auxiliary
feature_module=importlib.import_module(MEMBER['feature_module'])
Features=getattr(feature_module,{'r47':'RankCoordinateFeatures','r59':'DelayLogFeatures'}.get(VIEW,'ConditionalFeatures'))
def load_split(split):
    assert STAGE=='threefold' and json.loads((ROOT/'research_artifacts/r64_viewbag_fivefold/result.json').read_text())['eligible']
    df,raw,y,splits,_,tree,fold,_=baseline(split);best=np.load(ROOT/f'research_artifacts/r23_conditional_lean_{STAGE}/oof.npz')
    assert all(np.array_equal(best[k],v) for k,v in [('id',df.id),('target',y),('fold',fold)])
    return df,raw,y,splits,best['candidate'],tree,fold,auxiliary(df.id)

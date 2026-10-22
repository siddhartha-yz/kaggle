from research_r93_common import *

def repaired_plan():
 p=plan();r=json.loads((ROOT/'research_r93_repair_manifest.json').read_text());assert r['original_plan_sha256']==sha(ROOT/'research_r93_plan.json')
 for n,h in r['corrected_source_sha256'].items():assert sha(ROOT/n)==h,n
 assert sha(ROOT/'research_r93_membership_equivalence.json')==r['equivalence_result_sha256']
 return p,r

def checkpoint(name,compute):
 path=OUT/'checkpoints'/f'{name}.json';path.parent.mkdir(exist_ok=True);ph=sha(ROOT/'research_r93_plan.json');rh=sha(ROOT/'research_r93_repair_manifest.json')
 if path.exists():
  r=json.loads(path.read_text());assert r['status']=='complete' and r['plan_sha256']==ph and r['repair_sha256']==rh;print('REUSE_CHECKPOINT',name,flush=True);return r['data']
 tick=time.monotonic();value=compute();reserve();dump(path,dict(status='complete',plan_sha256=ph,repair_sha256=rh,completed_utc=now(),seconds=time.monotonic()-tick,data=value));print('SAVED_CHECKPOINT',name,flush=True);return value

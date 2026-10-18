"""Serial pre-test audit/replays then20context test inference; never submits."""
import subprocess,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;CPU=ROOT.parent/'.venv/bin/python'
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),**extra);(ROOT/'research_r36_post_state.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
def main():
    assert json.loads((ROOT/'research_r36_driver_state.json').read_text())['status']=='qualified_for_audit'
    assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True'
    if not (ROOT/'research_r36_final_inference_plan.json').exists():subprocess.run([str(CPU),'research_r36_freeze.py'],cwd=ROOT,check=True)
    state('metadata_audit')
    subprocess.run([str(CPU),'research_r36_audit.py'],cwd=ROOT,check=True)
    replay_records=[]
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for j in range(4):
                state('context_replay',stage=stage,fold=k,context=j)
                with open(ROOT/f'research_r36_replay_{stage}_{k}_{j}.log','w') as log:subprocess.run([sys.executable,'-u','research_r36_replay.py',stage,str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                r=json.loads((ROOT/f'research_artifacts/r36_replays/{stage}_fold{k}_context{j}.json').read_text());assert r['status']=='passed';replay_records.append(r)
    proof=[r['large32768_pointwise_FFN_bitwise_error'] for r in replay_records if r['large32768_pointwise_FFN_bitwise_error'] is not None];assert proof==[0.0]
    from research_r4_common import sha,now,dump
    meta=json.loads((ROOT/'research_r36_metadata_audit.json').read_text());assert meta['verified_context_records']==32
    audit=dict(meta,status='passed',completed_utc=now(),reconstructed_contexts=32,replay_records=replay_records,large32768_FFN_bitwise_error=0.0,metadata_audit_sha256=sha(ROOT/'research_r36_metadata_audit.json'),final_plan_sha256=sha(ROOT/'research_r36_final_inference_plan.json'));dump(ROOT/'research_r36_pre_inference_audit.json',audit)
    for k in range(5):
        for j in range(4):
            state('test_prediction',fold=k,context=j)
            with open(ROOT/f'research_r36_test_fold{k}_context{j}.log','w') as log:subprocess.run([sys.executable,'-u','research_r36_test_worker.py',str(k),str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    state('CSV_assembly');subprocess.run([str(CPU),'-u','research_r36_csv.py'],cwd=ROOT,check=True);state('validated_CSV_ready_for_Library',actual_top500_verified=False,new_formal_submissions=0)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error));raise

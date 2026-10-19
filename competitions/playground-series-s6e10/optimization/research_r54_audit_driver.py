"""Only after dual qualification: serial CPU source/maps audit and 32 GPU replays."""
import sys,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def main():
    for stage in ['fivefold','threefold']:assert json.loads((ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/result.json').read_text())['eligible']
    with open(ROOT/'research_r54_static_audit.log','w') as log:subprocess.run([str(CPU),'-u','research_r54_audit_static.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[]
    from hashlib import sha256
    sha=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    for stage,K in [('fivefold',5),('threefold',3)]:
        for k in range(K):
            for m in range(4):
                run=ROOT/f'research_artifacts/r54_tabpfn_v2_{stage}/fold{k}/context{m}'
                with (run/'independent_GPU_audit.log').open('w') as log:subprocess.run([sys.executable,'-u','research_r54_audit_gpu_worker.py',stage,str(k),str(m)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                r=json.loads((run/'independent_GPU_audit.json').read_text());assert r['status']=='passed';records.append(r)
    static=json.loads((ROOT/'research_r54_static_audit.json').read_text());assert static['status']=='passed_static_pending_GPU_replays' and len(records)==32
    result=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(ROOT/'research_r54_plan.json'),audit_plan_sha256=sha(ROOT/'research_r54_audit_plan.json'),static_audit_sha256=sha(ROOT/'research_r54_static_audit.json'),verified_contexts=32,independent_GPU_replays=records,validation=static['validation'],feature_generator_uses_official_SAT=False,source_original_safe_rows=129859,context_membership_and_maps_verified=True,positive_class='True=1',checkpoint_pretraining_provenance_file='research_r54_pretrained_provenance.json',cross_backend_bitwise_equivalence_claim=False,attribution='Built with PriorLabs-TabPFN',actual_top500_verified=False,limitations=static['limitations']);(ROOT/'research_r54_pre_inference_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('AUDIT_PASS',len(records),flush=True)
if __name__=='__main__':main()

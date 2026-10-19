"""Follow the existing validation process, recover only resource interruptions, then audit and export."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,subprocess,hashlib,re,shutil
from research_r72_headroom import wait_headroom
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
GPU=ROOT.parent/'.venv-tabicl-w3/bin/python'
MAX_RESOURCE_RETRIES=3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    for _ in range(20):
        try:return json.loads(Path(p).read_text())
        except json.JSONDecodeError:time.sleep(.2)
    raise RuntimeError('Repeated incomplete state JSON: '+str(p))
def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),parent_only_submitter=True,actual_top500_verified=False,**extra);(ROOT/'research_r72_followthrough_state.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
def own_alive(pid,script):
    try:return script in Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode(errors='replace') and Path(f'/proc/{int(pid)}/cwd').resolve()==ROOT
    except (OSError,ValueError):return False

def resource_failure(error,kind):
    pattern=r"research_r72_"+('worker' if kind=='validation' else 'audit_worker' if kind=='audit' else 'test_worker')+r"\.py', '([^']+)'(?:, '([^']+)')?(?:, '([^']+)')?"
    match=re.search(pattern,error)
    if match is None:return None
    args=[v for v in match.groups() if v is not None]
    if kind=='validation':stage,k,j=args;log=ROOT/f'research_r72_{stage}_fold{k}_context{j}.log'
    elif kind=='audit':stage,k,j=args;log=ROOT/f'research_r72_audit_{stage}_fold{k}_context{j}.log'
    else:k,j=args;log=ROOT/f'research_r72_test_fold{k}_context{j}.log'
    content=log.read_text(errors='replace') if log.exists() else ''
    is_guard='CalledProcessError(86,' in error
    is_headroom='AssertionError' in content and 'torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CAP+MIN_GPU_FREE' in content
    return dict(log=str(log),error=error,reason='taskresourceguard' if is_guard else 'initialheadroomcheck',args=args) if is_guard or is_headroom else None

def preserve_failure(kind,number,record):
    d=ROOT/f'research_artifacts/r72_control_failures/{kind}_retry{number}';d.mkdir(parents=True,exist_ok=True);(d/'record.json').write_text(json.dumps(record,indent=2)+'\n');log=Path(record['log'])
    if log.exists():shutil.copyfile(log,d/'original_failure.log')

def run_phase(label,program,python,kind=None):
    for attempt in range(MAX_RESOURCE_RETRIES+1):
        state(label,resource_retry=attempt)
        with open(ROOT/f'research_r72_{label}_attempt{attempt}.log','w') as log:p=subprocess.run([str(python),'-u',program],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode==0:return
        if kind is None:raise RuntimeError(label+' failed; inspect saved log')
        r=read(ROOT/('research_r72_audit_state.json' if kind=='audit' else 'research_r72_test_state.json'));failure=resource_failure(r.get('error',''),kind)
        if failure is None or attempt==MAX_RESOURCE_RETRIES:raise RuntimeError(label+' failed: '+str(r))
        preserve_failure(kind,attempt+1,failure);state('waiting_resources_before_same_phase_resume',phase=label,retry=attempt+1,reason=failure['reason']);wait_headroom(ROOT/f'research_r72_{kind}_retry_headroom.json')

def main():
    plan=read(ROOT/'research_r72_plan.json');ap=read(ROOT/'research_r72_audit_plan.json');assert ap['main_plan_sha256']==sha(ROOT/'research_r72_plan.json')
    for n,h in {**plan['source_sha256'],**plan['artifact_sha256'],**ap['additional_sha256']}.items():assert sha(ROOT/n)==h,n
    state('monitoring_existing_validation',original_driver_pid=1001611,original_session=15214);retries=0
    while True:
        r=read(ROOT/'research_r72_driver_state.json');status=r['status']
        if status in ['qualified_for_audit','not_qualified']:break
        if status=='failed':
            failure=resource_failure(r.get('error',''),'validation')
            if failure is None or retries>=MAX_RESOURCE_RETRIES:raise RuntimeError('Validationfailed without safe resource-only recovery: '+str(r))
            retries+=1;preserve_failure('validation',retries,failure);state('waiting_resources_before_frozen_validation_resume',retry=retries,reason=failure['reason']);wait_headroom(ROOT/'research_r72_validation_retry_headroom.json')
            with open(ROOT/f'research_r72_control_validation_resume{retries}.log','w') as log:subprocess.run([str(GPU),'-u','research_r72_driver.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            continue
        if not own_alive(r.get('driver_pid'), 'research_r72_driver.py'):
            last=read(ROOT/'research_r72_state.json')
            if own_alive(last.get('pid'), 'research_r72_worker.py'):time.sleep(5);continue
            raise RuntimeError('Validationdriverdisappeared withoutterminal state; preservecheckpoints and inspect beforeresume')
        time.sleep(5)
    if status=='not_qualified':state('not_qualified',stage=r.get('stage'),active_work=False);return
    prior=ROOT/'research_r72_pre_inference_audit.json'
    if prior.exists():
        a=read(prior);assert a['status']=='passed' and a['main_plan_sha256']==sha(ROOT/'research_r72_plan.json') and a['audit_plan_sha256']==sha(ROOT/'research_r72_audit_plan.json')
    else:run_phase('independent_audit','research_r72_audit_driver.py',GPU,kind='audit')
    run_phase('freeze_prediction','research_r72_prepare_prediction.py',CPU)
    run_phase('saved_context_test_inference','research_r72_test_driver.py',GPU,kind='test')
    run_phase('freeze_CSV_export','research_r72_prepare_export.py',CPU)
    run_phase('CSV_serialization','research_r72_csv.py',CPU)
    run_phase('independent_CSV_verification','research_r72_verify_csv.py',CPU)
    delivery=read(ROOT/'research_r72_delivery_audit.json');assert delivery['status']=='passed';state('checked_CSV_ready_for_Library',csv=delivery['csv'],sha256=delivery['sha256'],rows=delivery['rows'],active_work=False)
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='16'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    try:main()
    except BaseException as e:state('failed',error=repr(e),active_work=False);raise

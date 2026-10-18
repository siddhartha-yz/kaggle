"""Resource-only fresh-process retries of UNCHANGED frozen driver/worker.
No partial AUC inspection, no hyperparameter changes. Retry ONLY fresh recorded
allocator OOM with newly completed folds; stop on no progress or RAM/GPU guards.
"""
from pathlib import Path
import subprocess,json,time,datetime,shutil,hashlib,sys
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'.venv-gpu/bin/python'
def completed():
    result=[]
    for stage in ['fivefold','threefold']:
        for path in (ROOT/f'research_artifacts/r26_conditional_lean_{stage}').glob('fold*/result.json'):
            if json.loads(path.read_text()).get('status')=='complete':result.append(f'{stage}/{path.parent.name}')
    return sorted(result)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    history=[];last=completed()
    for attempt in range(1,9):
        for name in ['research_r26_fivefold.log','research_r26_threefold.log','research_r26_failure.json']:
            p=ROOT/name
            if p.exists():shutil.copy2(p,ROOT/f'research_r26_recovery_private_{attempt}_{name}')
        started=time.time()
        with open(ROOT/f'research_r26_recovery_driver_attempt{attempt}.log','w') as log:
            rc=subprocess.run([str(PYTHON),'-u','research_r26_driver.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        after=completed();r=dict(attempt=attempt,ended_utc=now(),exit_code=rc,completed_models=after,new_completed_models=sorted(set(after)-set(last)),frozen_training_code_and_seed_unchanged=True)
        history.append(r);record=dict(updated_utc=now(),resource_only=True,recovery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),attempts=history);(ROOT/'research_r26_recovery_log.json').write_text(json.dumps(record,indent=2)+'\n')
        if rc==0:print('RECOVERY_DRIVER_COMPLETE',json.loads((ROOT/'research_r26_driver_state.json').read_text())['status'],flush=True);return
        failures=list((ROOT/'research_artifacts').glob('r26_conditional_lean_*/resource_failure.json'))
        assert not any(p.stat().st_mtime>=started for p in failures),'Resourceguard fired: do not retry.'
        failure=ROOT/'research_r26_failure.json';assert failure.exists() and failure.stat().st_mtime>=started,'No fresh exception record: do not retry.'
        error=json.loads(failure.read_text())['error'];assert error.startswith('OutOfMemoryError('),'Nonallocator failure: do not retry.'
        assert set(after)>set(last),'No completed-model progress: do not retry.'
        last=after;print('RESOURCE_ONLY_RESTART',attempt,'completed',len(after),flush=True)
    raise RuntimeError('Resource-only retry budget exhausted; keep all completed models.')
if __name__=='__main__':main()

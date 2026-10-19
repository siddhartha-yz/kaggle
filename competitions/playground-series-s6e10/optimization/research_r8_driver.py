"""Run both fixed fivefold routes, then confirm the eligible winner once."""
import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent

def now(): return datetime.now(timezone.utc).isoformat()
def state(status, **extra):
    record = dict(updated_utc=now(), status=status, driver_pid=os.getpid(),
                  whole_top100_task_completed=False, formal_used=1, remaining=1, **extra)
    (ROOT/'research_r8_driver_state.json').write_text(json.dumps(record,indent=2)+'\n')
def run(script, args, name):
    with open(ROOT/name, 'w') as handle:
        subprocess.run([sys.executable, str(ROOT/script), *args], cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)

def main():
    for variant in ('lean', 'append'):
        state('training', stage='fivefold', variant=variant, active_training=True)
        run('research_r8_worker.py', [variant], f'research_r8_{variant}_fivefold.log')
    # Neither route is scored until both fixed fivefold runs are complete.
    from research_r8_check import check
    results = [check(variant, 'fivefold') for variant in ('lean', 'append')]
    eligible = [r for r in results if r['eligible']]
    if not eligible:
        state('not_qualified', stage='fivefold', active_training=False)
        return
    winner = max(eligible, key=lambda r:r['vs_private_half']['pooled']['candidate_auc'])['variant']
    (ROOT/'research_r8_selected.json').write_text(json.dumps(dict(
        frozen_utc=now(), variant=winner, selection='highest fivefold AUC among dual-baseline eligible routes'),indent=2)+'\n')
    state('training', stage='threefold', variant=winner, active_training=True)
    run('research_r8_worker.py', [winner,'--threefold'], f'research_r8_{winner}_threefold.log')
    result = check(winner, 'threefold')
    state('qualified_for_audit' if result['eligible'] else 'not_qualified',
          stage='threefold', variant=winner, active_training=False)

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=repr(error), active_training=False)
        raise

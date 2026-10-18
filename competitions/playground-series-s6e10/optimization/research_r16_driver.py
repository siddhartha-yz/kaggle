"""Run both fixed GPU views before scoring, then confirm eligible winner once."""
import os,sys,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def state(status,**extra):
    x=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),formal_used=1,
        remaining=1,whole_top100_task_completed=False,**extra);(ROOT/'research_r16_driver_state.json').write_text(json.dumps(x,indent=2)+'\n')
def run(script,args,filename):
    with open(ROOT/filename,'w') as log:subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    for variant in ['numeric57','nominal58']:
        state('training',variant=variant,stage='fivefold',active_training=True)
        run('research_r16_worker.py',[variant],f'research_r16_{variant}_fivefold.log')
    from research_r16_check import check
    results=[check(v,'fivefold') for v in ['numeric57','nominal58']]
    eligible=[r for r in results if r['eligible']]
    if not eligible:state('not_qualified',stage='fivefold',active_training=False);return
    winner=max(eligible,key=lambda r:r['vs_private_v9']['pooled']['candidate_auc'])['variant']
    (ROOT/'research_r16_selected.json').write_text(json.dumps(dict(variant=winner,frozen_utc=datetime.now(timezone.utc).isoformat(),rule='highestfivefoldauc amongeligibleviews'),indent=2)+'\n')
    state('training',variant=winner,stage='threefold',active_training=True)
    run('research_r16_worker.py',[winner,'--threefold'],f'research_r16_{winner}_threefold.log')
    result=check(winner,'threefold')
    state('qualified_for_audit' if result['eligible'] else 'not_qualified',variant=winner,stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

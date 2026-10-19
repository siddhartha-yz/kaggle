"""Complete nested GPU tree-count algorithm fivefold before scoring, then confirm once."""
import os,sys,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def state(status,**extra):
    x=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,driver_pid=os.getpid(),formal_used=1,
        remaining=1,whole_top100_task_completed=False,**extra);(ROOT/'research_r18_driver_state.json').write_text(json.dumps(x,indent=2)+'\n')
def run(script,args,filename):
    with open(ROOT/filename,'w') as log:subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    from research_r18_check import check
    variant='numeric57'
    for stage,args in [('fivefold',[]),('threefold',['--threefold'])]:
        state('training',variant=variant,stage=stage,active_training=True)
        run('research_r18_worker.py',[variant,*args],f'research_r18_{variant}_{stage}.log')
        result=check(variant,stage)
        if not result['eligible']:
            state('not_qualified',stage=stage,active_training=False);return
    state('qualified_for_audit',variant=variant,stage='threefold',active_training=False)
if __name__=='__main__':
    try:main()
    except BaseException as error:state('failed',error=repr(error),active_training=False);raise

"""Follow existing validation into independent audit and inference only if both gates pass."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,subprocess
ROOT=Path(__file__).resolve().parent
GPU=ROOT.parent/'.venv-gpu/bin/python'
CPU=ROOT.parent/'.venv/bin/python'

def state(status,**extra):
    r=dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),actual_top500_verified=False,**extra);(ROOT/'research_r68_followthrough_state.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
def main():
    state('waiting_existing_validation',driver_pid=966223)
    while True:
        try:r=json.loads((ROOT/'research_r68_driver_state.json').read_text())
        except json.JSONDecodeError:time.sleep(1);continue
        if r['status'] in ['qualified_for_audit','not_qualified','failed']:break
        os.kill(966223,0);time.sleep(5)
    if r['status']!='qualified_for_audit':state(r['status']);return
    for label,program,python in [('independent_audit','research_r68_audit.py',GPU),('freeze_final_inference','research_r68_prepare_final.py',CPU),('bounded_saved_model_inference','research_r68_csv.py',GPU)]:
        state(label)
        with open(ROOT/f'research_r68_{label}.log','w') as log:subprocess.run([str(python),'-u',program],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    state('checked_CSV_ready_for_Library')
if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise

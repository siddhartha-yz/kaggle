"""No fitting of new weights; serialized contexts, sequential bounded final inference."""
import sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CPU=ROOT.parent/'.venv/bin/python'
def main():
    assert json.loads((ROOT/'research_r54_pre_inference_audit.json').read_text())['status']=='passed'
    for k in range(5):
        for m in range(4):
            with open(ROOT/f'research_r54_test_fold{k}_context{m}.log','a') as log:subprocess.run([sys.executable,'-u','research_r54_csv_worker.py',str(k),str(m)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(CPU),'-u','research_r54_csv_assemble.py'],cwd=ROOT,check=True)
if __name__=='__main__':main()

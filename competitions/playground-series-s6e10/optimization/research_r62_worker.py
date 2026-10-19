"""Budget-guardedCPU6stageforthe97fieldoriginalclasscandidate."""
from research_r40_worker import ROOT,watch_cpu
import threading,sys
from research_r62_worker_body import main
if __name__=='__main__':
    peak=[0];stop=threading.Event();t=threading.Thread(target=watch_cpu,args=(stop,peak,ROOT/f'research_r62_{sys.argv[1]}_resource_failure.json'),daemon=True);t.start()
    try:main(sys.argv[1])
    finally:stop.set();t.join(timeout=5)

"""Wait for measured laptop headroom before an unchanged serial model reconstruction."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,subprocess
ROOT=Path(__file__).resolve().parent

def snapshot():
    available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8);assert p.returncode==0,p.stderr;free=int(p.stdout.strip().splitlines()[0])*2**20
    return dict(available_RAM_bytes=available,free_GPU_bytes=free,loadavg=list(os.getloadavg()))

def wait_headroom(path):
    tick=time.monotonic();consecutive=0
    while True:
        info=snapshot();enough=info['available_RAM_bytes']>=12*1024**3 and info['free_GPU_bytes']>=int(5.5*1024**3);consecutive=consecutive+1 if enough else 0
        if consecutive>=2:return info
        record=dict(status='waiting_measured_headroom',updated_utc=datetime.now(timezone.utc).isoformat(),wait_seconds=time.monotonic()-tick,**info);Path(path).write_text(json.dumps(record,indent=2)+'\n');time.sleep(5)

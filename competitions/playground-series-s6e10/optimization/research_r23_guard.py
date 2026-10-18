"""Fail-closed monitor for authorized local GPU/RAM limits, never touches peers."""
import os,json,subprocess,psutil
from pathlib import Path

def watch(stop,peak,path,context=None):
    def fail(**fields):
        Path(path).write_text(json.dumps(dict(context or {},**fields),indent=2)+'\n');os._exit(86)
    try:
        process=psutil.Process()
        while not stop.wait(.25):
            q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
            for line in q.stdout.splitlines():
                try:pid,memory=[int(v.strip()) for v in line.split(',')]
                except ValueError:continue
                if pid==os.getpid():peak[0]=max(peak[0],memory*1024**2)
            available=int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024;rss=process.memory_info().rss
            if peak[0]>1073741824 or available<2*1024**3 or rss>8*1024**3:fail(status='resource_guard_stopped',peak_gpu_process_bytes=peak[0],available_RAM=available,RSS=rss)
    except BaseException as error:fail(status='resource_monitor_failed',error=repr(error))

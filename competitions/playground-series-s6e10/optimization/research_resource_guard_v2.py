"""Task-only budget enforcement with reserved desktop/host headroom; no other process mutation."""
import os,time,json,subprocess,pathlib,resource
CPU_THREADS=16
CAP=3221225472
CEILING=3758096384
RSS_CEILING=8589934592
MIN_AVAILABLE=4294967296
MIN_GPU_FREE=2147483648

def host_available():
    return int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable:')))*1024

def watch(stop,peak,path):
    while not stop.wait(.25):
        reason=None
        try:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4);assert p.returncode==0
            own=0
            for l in p.stdout.splitlines():
                fields=l.split(',')
                if len(fields)==2 and fields[0].strip()==str(os.getpid()):own+=int(fields[1].strip())*2**20
            peak[0]=max(peak[0],own)
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4);assert gpu.returncode==0;free=int(gpu.stdout.strip().splitlines()[0])*2**20
            rss=int(pathlib.Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');available=host_available()
            if own>CEILING:reason='own GPU process ceiling'
            elif rss>RSS_CEILING:reason='own RSS ceiling'
            elif available<MIN_AVAILABLE:reason='host available RAM reserve'
            elif free<MIN_GPU_FREE:reason='desktop GPU free reserve'
            if reason:
                pathlib.Path(path).write_text(json.dumps(dict(reason=reason,own_GPU_bytes=own,peak_GPU_bytes=peak[0],RSS_bytes=rss,host_available_bytes=available,free_GPU_bytes=free)));os._exit(86)
        except BaseException as e:
            pathlib.Path(path).write_text(json.dumps(dict(reason='resource telemetry unavailable',error=repr(e))));os._exit(86)

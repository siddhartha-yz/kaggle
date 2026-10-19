"""Absolute user stop policy; no automatic resume, model-specific normal exit."""
from pathlib import Path
from datetime import datetime,timezone
import json
ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'research_run_deadline.json').read_text())
SOFT=datetime.fromisoformat(POLICY['training_stop_utc']).timestamp()
HARD=datetime.fromisoformat(POLICY['hard_stop_utc']).timestamp()
class StopByDeadline(Exception):pass
def utc_seconds():return datetime.now(timezone.utc).timestamp()
def check_training():
 if utc_seconds()>=SOFT:raise StopByDeadline('User15:15UTCtrainingcutoff;preservepartialartifacts,noautomaticresume')
def check_finalization():
 if utc_seconds()>=HARD-120:raise StopByDeadline('User15:59UTChardstop;finalizationstops2minbeforedeadline')

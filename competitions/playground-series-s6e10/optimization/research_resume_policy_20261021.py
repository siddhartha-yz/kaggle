"""User-authorized reopened campaign; no obsolete calendar stop condition."""
from pathlib import Path
from datetime import datetime,timezone
import json
ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'research_r79r2_run_policy.json').read_text())
HARD=float('inf')
SOFT=float('inf')
class StopByDeadline(Exception):pass
def utc_seconds():return datetime.now(timezone.utc).timestamp()
def check_training():
 if (ROOT/POLICY['stop_request_marker']).exists():raise StopByDeadline('Explicitstoprequestedforthisreopenedrun')
def check_finalization():check_training()

"""Record the macOS GPU counters beside a timed session log; stop at its summary."""
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from datetime import datetime, timezone

folder = Path(sys.argv[1])
with (folder / "gpu.jsonl").open("x", buffering=1) as out:
    while True:
        t = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            stats = plistlib.loads(subprocess.check_output(
                ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"], timeout=4))[0]["PerformanceStatistics"]
            row = {"utc": t, "device_pct": stats["Device Utilization %"],
                   "device_bytes": stats.get("In use system memory")}
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as e:
            row = {"utc": t, "error": type(e).__name__}
        out.write(json.dumps(row) + "\n")
        with (folder / "events.jsonl").open("rb") as events:
            events.seek(max(0, events.seek(0, 2) - 2048))
            if b'"event": "summary"' in events.read():
                break
        time.sleep(5)

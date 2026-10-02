"""Record everything needed to reproduce a run (Verifiability + winner obligations).

    from dspo.runlog import record_run
    record_run("runs/manifest.jsonl", config={"arm": "D", "budget": 100}, seed=0)
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import socket
import subprocess
from pathlib import Path


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    repo = Path(__file__).resolve().parent.parent
    return {
        "utc_time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _cmd("git", "-C", str(repo), "rev-parse", "HEAD"),
        "git_dirty": bool(_cmd("git", "-C", str(repo), "status", "--porcelain")),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpus": _cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"),
    }


def record_run(manifest: str | Path, config: dict, seed: int, **extra) -> dict:
    """Append one JSON line describing a run to `manifest` and return it."""
    entry = {"config": config, "seed": seed, **environment(), **extra}
    path = Path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry

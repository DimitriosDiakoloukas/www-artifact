"""The replication gate: a run or measurement into replication/ starts only if replication/LOCK.json
exists, the protocol and the replication code match the hashes it records, the guarded paths are
clean and HEAD descends from the lock commit."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "replication"
PROTOCOL = HERE / "PROTOCOL.md"
LOCK = HERE / "LOCK.json"
CODE = ("gate.py", "lock.py", "run_chain_r.py", "run_queue_r.py", "measure.py", "analyze.py", "make_jobs.py", "run_all.sh",
        "jobs.txt")
GUARDED = ("src", "scripts", "exploratory/variants.py", *(f"replication/{c}" for c in CODE),
           "replication/PROTOCOL.md", "replication/LOCK.json")


def sha256_file(p) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def check_lock() -> dict:
    if not LOCK.exists():
        raise SystemExit("replication run refused: no replication/LOCK.json")
    lock = json.loads(LOCK.read_text())
    if sha256_file(PROTOCOL) != lock["protocol_sha256"]:
        raise SystemExit("replication run refused: the protocol changed after the lock")
    for c in CODE:
        if c != "jobs.txt" and sha256_file(HERE / c) != lock["code_sha256"][c]:
            raise SystemExit(f"replication run refused: replication/{c} changed after the lock")
    dirty = _git("status", "--porcelain", "--", *GUARDED).stdout.strip()
    if dirty:
        raise SystemExit(f"replication run refused: uncommitted changes in guarded paths:\n{dirty}")
    if _git("merge-base", "--is-ancestor", lock["commit"], "HEAD").returncode != 0:
        raise SystemExit("replication run refused: HEAD does not descend from the lock commit")
    return lock

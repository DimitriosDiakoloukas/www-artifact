"""Locks the confirmatory protocol: run after CONFIRMATORY_PROTOCOL.md and the selection are committed,
then commit the PROTOCOL_LOCK.json it writes. Refuses a dirty tree or a protocol still marked DRAFT
or holding [AT FREEZE] items.

  python3 scripts/lock_protocol.py --selection development/20261003-phase1-selection/selection.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange.lock import GUARDED, LOCK, PROTOCOL  # noqa: E402
from srange.paths import REPO  # noqa: E402
from srange.provenance import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    a = ap.parse_args()
    text = PROTOCOL.read_text()
    if "DRAFT" in text.splitlines()[0] or "[AT FREEZE" in text:
        raise SystemExit("protocol still a draft or has unfilled [AT FREEZE] items")
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", *GUARDED[:-1], a.selection):
        raise SystemExit("commit the protocol, code and selection before locking")
    if LOCK.exists():
        raise SystemExit("already locked")
    sel = json.loads((REPO / a.selection).read_text())
    lock = {"protocol_sha256": sha256_file(PROTOCOL), "commit": git("rev-parse", "HEAD"),
            "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_file": a.selection, "selection_sha256": sel["sha256"]}
    LOCK.write_text(json.dumps(lock, indent=1) + "\n")
    print(json.dumps(lock, indent=1))
    print("now commit confirmatory/PROTOCOL_LOCK.json; confirmatory runs check it")


if __name__ == "__main__":
    main()

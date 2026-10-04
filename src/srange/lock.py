"""The confirmatory gate: a run in confirmatory/ starts only if the protocol is locked and unchanged,
the code is clean, and HEAD descends from the lock commit."""
from __future__ import annotations

import json
import subprocess

from srange.paths import CONFIRMATORY, REPO
from srange.provenance import sha256_file

PROTOCOL = CONFIRMATORY / "CONFIRMATORY_PROTOCOL.md"
LOCK = CONFIRMATORY / "PROTOCOL_LOCK.json"
GUARDED = ("src", "scripts", "tests", "confirmatory/CONFIRMATORY_PROTOCOL.md", "confirmatory/PROTOCOL_LOCK.json")


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def check_lock() -> dict:
    if not LOCK.exists():
        raise SystemExit("confirmatory run refused: no PROTOCOL_LOCK.json")
    lock = json.loads(LOCK.read_text())
    if sha256_file(PROTOCOL) != lock["protocol_sha256"]:
        raise SystemExit("confirmatory run refused: the protocol changed after the lock")
    dirty = _git("status", "--porcelain", "--", *GUARDED).stdout.strip()
    if dirty:
        raise SystemExit(f"confirmatory run refused: uncommitted changes in guarded paths:\n{dirty}")
    if _git("merge-base", "--is-ancestor", lock["commit"], "HEAD").returncode != 0:
        raise SystemExit("confirmatory run refused: HEAD does not descend from the lock commit")
    return lock


def frozen_selection(lock: dict) -> dict:
    sel = json.loads((REPO / lock["selection_file"]).read_text())
    if sel["sha256"] != lock["selection_sha256"]:
        raise SystemExit("confirmatory run refused: the hyperparameter selection changed after the lock")
    return sel["selection"]

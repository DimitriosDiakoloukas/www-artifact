"""Hashes, checkpoints and run records (plan, Section H)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from srange.paths import REPO, STORE

PACKAGES = ("torch", "torch-geometric", "torch-geometric-signed-directed", "numpy", "scipy", "scikit-learn")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_hash(state: dict) -> str:
    """Hash of tensor contents, independent of pickle layout (vendored from the old stage study)."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(array_sha256(value.detach().cpu().numpy()).encode())
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "HEAD") or None
    dirty = bool(run("status", "--porcelain", "--", "src", "scripts", "configs", "tests"))
    return {"git_commit": commit, "git_dirty": dirty}


def source_tree_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "src").rglob("*.py")):
        h.update(str(p.relative_to(REPO)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def environment() -> dict:
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            vers[p] = None
    return {"python": platform.python_version(), "packages": vers, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def save_checkpoint(obj: dict) -> tuple[str, str]:
    """Write to the content-addressed store; returns (sha256, path)."""
    (STORE / "objects").mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STORE / "objects", suffix=".tmp")
    os.close(fd)
    torch.save(obj, tmp)
    sha = sha256_file(tmp)
    dst = STORE / "objects" / f"{sha}.pt"
    os.replace(tmp, dst)
    return sha, str(dst)


def load_checkpoint(sha: str, map_location="cpu") -> dict:
    path = STORE / "objects" / f"{sha}.pt"
    got = sha256_file(path)
    if got != sha:
        raise ValueError(f"checkpoint {path} hashes to {got}, expected {sha}")
    return torch.load(path, map_location=map_location, weights_only=False)


def write_record(path, rec: dict) -> str:
    """Adds result_sha256 (hash of the canonical record without it) and writes atomically."""
    rec = dict(rec)
    rec.pop("result_sha256", None)
    rec["result_sha256"] = sha256_bytes(canonical(rec).encode())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, sort_keys=True, indent=1, allow_nan=False))
    os.replace(tmp, path)
    return rec["result_sha256"]


def verify_record(path) -> dict:
    rec = json.loads(Path(path).read_text())
    want = rec.pop("result_sha256")
    if sha256_bytes(canonical(rec).encode()) != want:
        raise ValueError(f"{path}: result_sha256 mismatch")
    rec["result_sha256"] = want
    return rec


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

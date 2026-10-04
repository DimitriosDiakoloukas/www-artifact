"""Edge splits, written to disk once and read back, never regenerated silently."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FRACTIONS = (0.6, 0.2, 0.2)


def make_split(E: int, seed: int) -> dict:
    perm = np.random.default_rng(seed).permutation(E)
    n_tr, n_va = int(FRACTIONS[0] * E), int(FRACTIONS[1] * E)
    return {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va:])}


def split_sha256(split: dict) -> str:
    h = hashlib.sha256()
    for k in ("train", "val", "test"):
        h.update(k.encode())
        h.update(np.ascontiguousarray(split[k].astype(np.int64)).tobytes())
    return h.hexdigest()


def load_or_make_split(root: Path, dataset: str, processed_sha: str, E: int, seed: int) -> tuple[dict, str]:
    """Split indices for (dataset, seed); created on first use, then verified on every read."""
    d = Path(root) / "splits" / dataset / f"seed{seed}"
    meta_p = d / "split.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["processed_sha256"] != processed_sha:
            raise ValueError(f"split {d} was made for a different processed graph")
        split = {k: np.load(d / f"{k}.npy") for k in ("train", "val", "test")}
        if split_sha256(split) != meta["split_sha256"]:
            raise ValueError(f"split {d} does not match its recorded hash")
        return split, meta["split_sha256"]
    split = make_split(E, seed)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in split.items():
        np.save(d / f"{k}.npy", v)
    sha = split_sha256(split)
    meta_p.write_text(json.dumps({"dataset": dataset, "seed": seed, "fractions": FRACTIONS,
                                  "processed_sha256": processed_sha, "split_sha256": sha}, indent=1))
    return split, sha

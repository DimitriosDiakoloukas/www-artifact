"""Input features. Spectral features summarise global structure; random features carry none."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def spectral_features(n, tr_edges, tr_signs, dim=64, svd_seed=0) -> torch.Tensor:
    """PyG's signed spectral features (truncated SVD of the training signed adjacency), vendored from
    the old harness: the SVD seed is fixed and the caller's RNG state is preserved."""
    from torch_geometric.nn import SignedGCN
    e = torch.as_tensor(tr_edges, dtype=torch.long).t()
    pos, neg = e[:, tr_signs > 0], e[:, tr_signs < 0]
    state = np.random.get_state()
    try:
        np.random.seed(svd_seed)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            m = SignedGCN(dim, dim, num_layers=2)
            return m.create_spectral_features(pos, neg, num_nodes=n).float()
    finally:
        np.random.set_state(state)


def cached_spectral_features(store, key: str, n, tr_edges, tr_signs, dim=64, svd_seed=0) -> torch.Tensor:
    """spectral_features, computed once per (dataset, split, dim, seed) and verified on every read."""
    import json
    from pathlib import Path
    d = Path(store) / "features"
    d.mkdir(parents=True, exist_ok=True)
    f, m = d / f"{key}-spectral{dim}-svd{svd_seed}.pt", d / f"{key}-spectral{dim}-svd{svd_seed}.json"
    if f.exists() and m.exists():
        X = torch.load(f)
        if tensor_sha256(X) != json.loads(m.read_text())["sha256"]:
            raise ValueError(f"cached features {f} do not match their hash")
        return X
    X = spectral_features(n, tr_edges, tr_signs, dim, svd_seed)
    tmp = f.with_suffix(".tmp")
    torch.save(X, tmp); tmp.replace(f)
    m.write_text(json.dumps({"sha256": tensor_sha256(X)}))
    return X


def random_features(n, dim=64, seed=0) -> torch.Tensor:
    """Gaussian features with unit-norm columns, the scale of the spectral features."""
    g = torch.Generator().manual_seed(int(seed))
    X = torch.randn(n, dim, generator=g)
    return X / X.norm(dim=0, keepdim=True)


def tensor_sha256(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()

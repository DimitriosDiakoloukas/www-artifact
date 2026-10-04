"""One training procedure for every architecture: full batch, Adam, the same clipping for all,
checkpoint selected on validation AUC. Test labels never enter this module."""
from __future__ import annotations

import copy
import inspect
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@dataclass
class TrainConfig:
    lr: float = 5e-3
    weight_decay: float = 1e-5
    max_epochs: int = 300
    eval_every: int = 5
    patience: int = 20            # evaluations without improvement
    clip: float = 1.0             # applied to every architecture
    seed: int = 0

    def as_dict(self):
        return asdict(self)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weighted_bce(logits, y):
    pos = y.sum().clamp(min=1.0)
    pos_weight = ((y.numel() - y.sum()) / pos).reshape(1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def auc(y, scores) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def fit(modules: dict, loss_fn, val_fn, cfg: TrainConfig) -> dict:
    """modules: name -> nn.Module. loss_fn() -> training loss; val_fn() -> validation AUC (called in
    eval mode). Restores the best validation state and returns the training diagnostics."""
    params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_ep, best_state, since = -math.inf, 0, None, 0
    trace, flags = [], []
    grad_norms, clipped = [], 0
    t0 = time.time()
    for ep in range(1, cfg.max_epochs + 1):
        for m in modules.values():
            m.train()
        loss = loss_fn()
        if not torch.isfinite(loss):
            flags.append(f"nonfinite_loss_epoch_{ep}")
            break
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
        grad_norms.append(gn)
        clipped += gn > cfg.clip
        opt.step()
        if ep % cfg.eval_every == 0 or ep == cfg.max_epochs:
            for m in modules.values():
                m.eval()
            with torch.no_grad():
                v = val_fn()
            trace.append({"epoch": ep, "train_loss": float(loss.detach()), "val_auc": v})
            if np.isfinite(v) and v > best:
                best, best_ep, since = v, ep, 0
                best_state = {k: copy.deepcopy(m.state_dict()) for k, m in modules.items()}
            else:
                since += 1
                if since >= cfg.patience:
                    break
    if best_state is None:
        flags.append("no_valid_checkpoint")
    else:
        for k, m in modules.items():
            m.load_state_dict(best_state[k])
    for m in modules.values():
        m.eval()
    gn = np.asarray(grad_norms) if grad_norms else np.zeros(1)
    return {"best_val_auc": best if best_state is not None else None, "selected_epoch": best_ep,
            "epochs_run": ep, "trace": trace, "flags": flags, "seconds": time.time() - t0,
            "grad_norm": {"median": float(np.median(gn)), "max": float(gn.max())},
            "clip_events": int(clipped)}


assert "test" not in inspect.signature(fit).parameters

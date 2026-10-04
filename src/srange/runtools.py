"""Small helpers shared by the run scripts."""
from __future__ import annotations

import math

import numpy as np


def clean(x):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def truncation_schedule(T, max_skip):
    """Retained steps T, T/2, ..., 1, 0, as far as the architecture can remove steps."""
    kept, k = [], T
    while k >= 1:
        kept.append(k)
        k //= 2
    kept.append(0)
    return [T - k for k in kept if T - k <= max_skip]

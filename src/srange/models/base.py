"""Encoder interface shared by every architecture.

forward(X, g, skip=0) -> H. `skip` removes the earliest `skip` removable propagation steps of a
trained model (same-checkpoint truncation); the receptive field is then T - skip hops.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


class Encoder(nn.Module):
    """Set `memory_efficient = True` to recompute each propagation step in the backward pass instead of
    storing its per-edge tensors (identical results; needed at T = 32 on the largest graphs)."""
    arch: str = ""
    memory_efficient: bool = False

    def step(self, fn, *args):
        if self.memory_efficient and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    T: int = 0                 # propagation steps = receptive-field radius
    nominal_layers: int = 0
    out_dim: int = 0

    def max_skip(self) -> int:
        raise NotImplementedError

    def receptive_field(self, skip: int = 0) -> int:
        return self.T - skip

    def depth_record(self) -> dict:
        return {"arch": self.arch, "propagation_steps": self.T, "nominal_layers": self.nominal_layers,
                "max_skip": self.max_skip()}

    def check_skip(self, skip: int):
        if not 0 <= skip <= self.max_skip():
            raise ValueError(f"{self.arch}: skip={skip} outside [0, {self.max_skip()}]")

"""Message-passing graph shared by every encoder.

Edges are stored in both orientations; messages flow from edge_index[1] (source) into
edge_index[0] (target). Signs are a float tensor: encoders weight each edge by
w_pos = (1 + s) / 2 and w_neg = (1 - s) / 2, which equals the discrete formulation at s = +-1 and
makes sign interventions (flips) and sign gradients uniform across architectures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass
class MPGraph:
    n: int
    edge_index: torch.Tensor      # [2, 2E] (target, source)
    edge_id: torch.Tensor         # [2E] undirected edge id
    sign_und: torch.Tensor        # [E] float signs, one per undirected edge

    @property
    def E(self) -> int:
        return int(self.sign_und.numel())

    @property
    def sign(self) -> torch.Tensor:
        return self.sign_und[self.edge_id]

    @property
    def w_pos(self) -> torch.Tensor:
        return (1.0 + self.sign) * 0.5

    @property
    def w_neg(self) -> torch.Tensor:
        return (1.0 - self.sign) * 0.5

    def unsigned_degree(self) -> torch.Tensor:
        """Number of incident edges, independent of signs."""
        dev, dt = self.edge_index.device, self.sign_und.dtype
        return torch.zeros(self.n, dtype=dt, device=dev).index_add_(
            0, self.edge_index[0], torch.ones(self.edge_index.size(1), dtype=dt, device=dev))

    def with_signs(self, sign_und: torch.Tensor) -> "MPGraph":
        return MPGraph(self.n, self.edge_index, self.edge_id, sign_und)

    def flipped(self, f: int) -> "MPGraph":
        s = self.sign_und.clone()
        s[f] = -s[f]
        return self.with_signs(s)

    def to(self, device) -> "MPGraph":
        return MPGraph(self.n, self.edge_index.to(device), self.edge_id.to(device), self.sign_und.to(device))


def mp_graph(n: int, edges: np.ndarray, signs: np.ndarray, device="cpu", dtype=torch.float32) -> MPGraph:
    """Both orientations of each undirected edge (edges[:, 0] != edges[:, 1])."""
    e = torch.as_tensor(edges, dtype=torch.long)
    E = e.size(0)
    ei = torch.cat([e.t(), e.flip(1).t()], dim=1)               # [2, 2E]: (a<-b) then (b<-a)
    eid = torch.cat([torch.arange(E), torch.arange(E)])
    s = torch.as_tensor(signs, dtype=dtype)
    return MPGraph(n, ei.contiguous(), eid, s).to(device)


def hop_distances(n: int, edges: np.ndarray, sources) -> np.ndarray:
    """BFS hop distances [len(sources), n] on the undirected graph; inf where unreachable."""
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return shortest_path(A, unweighted=True, directed=False, indices=np.asarray(sources))

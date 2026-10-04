from srange.models.bgsd import BGSD
from srange.models.sgcn import SGCN
from srange.models.sidnet import SIDNET
from srange.models.slgnn import SLGNN

ARCHS = {"SGCN": SGCN, "SLGNN": SLGNN, "SIDNET": SIDNET, "BGSD": BGSD}


def build(arch: str, in_dim: int, T: int, hidden: int = 64, **knobs):
    """T is the number of propagation steps; each class maps it to its native hyperparameters
    (SIDNET: L = 2 layers of K = T / 2 diffusion steps)."""
    return ARCHS[arch](in_dim, hidden=hidden, T=T, **knobs)

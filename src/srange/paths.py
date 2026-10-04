"""Campaign locations. Checkpoints and raw data live outside git, in a content-addressed store."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("SRANGE_STORE", Path.home() / "signed-range-store"))
DEVELOPMENT = REPO / "development"
CONFIRMATORY = REPO / "confirmatory"

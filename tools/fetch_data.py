"""Downloads the six SNAP networks into $SRANGE_STORE/raw and checks each against its pinned SHA-256.

  SRANGE_STORE=<dir> python3 tools/fetch_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange.data.snap import SNAP, load_snap  # noqa: E402

for name in SNAP:
    ds = load_snap(name, download=True)
    print(f"{name}: {ds.n} users, {len(ds.edges)} relations, raw and processed hashes verified")

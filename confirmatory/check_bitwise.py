"""Bitwise round trip of confirmatory checkpoints: the unchanged scripts/check_study.py round trip, run under
the same deterministic kernels as the runs themselves (the locked checker does not switch them on).

  python3 confirmatory/check_bitwise.py <n> [--device cuda:0]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import check_study  # noqa: E402
from srange.provenance import verify_record  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
runs = sorted(p for p in (ROOT / "confirmatory/native/runs").glob("*.json") if not p.name.startswith("local-"))
sample = runs[:: max(1, len(runs) // n)][:n]
bitwise = 0
for p in sample:
    rec = verify_record(p)
    dv, dl, same = check_study.roundtrip(rec, "cuda:0")
    bitwise += same
    print(f"{rec['run_id']}: |dval|={dv:.2e} max|dlogit|={dl:.2e} bitwise={same}", flush=True)
print(f"RESULT {'PASS' if bitwise == len(sample) else 'FAIL'}: {bitwise}/{len(sample)} bitwise")

"""Plain test runner (no pytest in the shared environment): python3 tests/run.py [module ...]."""
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

mods = sys.argv[1:] or sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
failed = 0
for name in mods:
    mod = importlib.import_module(f"tests.{name}")
    for t in mod.TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {name}.{t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}.{t.__name__}")
            traceback.print_exc()
print("RESULT", "PASS" if failed == 0 else f"FAIL ({failed})")
sys.exit(1 if failed else 0)

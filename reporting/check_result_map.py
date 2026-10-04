"""Submission gate 4: every file in the paper's generated/ and figures/ matches the latest SHA-256 that
RESULT_MAP.md records for it.

  python3 reporting/check_result_map.py --paper <paper>
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    paper = Path(ap.parse_args().paper).expanduser()
    latest = {}
    for line in (REPO / "RESULT_MAP.md").read_text().splitlines():
        for f, h in re.findall(r"`((?:generated|figures)/[^`]+)` \(`([0-9a-f]{64})`\)", line):
            latest[f] = h
    files = sorted(p for sub, pat in (("generated", "*"), ("figures", "*.pdf")) for p in (paper / sub).rglob(pat)
                   if p.is_file() and p.suffix in (".tex", ".json", ".pdf"))
    bad = []
    for p in files:
        f = str(p.relative_to(paper))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if latest.get(f) != h:
            bad.append(f"{'missing' if f not in latest else 'stale'}: {f} ({h})")
    print("\n".join(bad) or f"all {len(files)} outputs match RESULT_MAP")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

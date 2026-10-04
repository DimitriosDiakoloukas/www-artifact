#!/bin/bash
# After replication/run_all.sh has written its analysis: regenerate every paper asset and rebuild the PDF.
# Commits nothing; the outputs are reviewed first.
#   bash reporting/finish.sh <paper>
set -eu
cd "$(dirname "$0")/.."
PAPER=${1:-$HOME/workspace/papers/www2027}
PY=$HOME/workspace/.venv/bin/python
until grep -q "analysis written" replication/driver.log 2>/dev/null; do sleep 60; done
$PY replication/analyze.py --out "$PAPER/generated" | tail -3
cmp replication/analysis/replication.json "$PAPER/generated/replication.json" && echo "replication.json identical to the driver's"
$PY reporting/sign_mass.py --paper "$PAPER" | tail -1
$PY reporting/protocol_assets.py --paper "$PAPER" | tail -2
$PY reporting/paper_assets.py --paper "$PAPER"
cd "$PAPER" && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null; \
  pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null
$PY -c "import fitz; d=fitz.open('main.pdf'); print('pages', len(d), 'PENDING left', sum('PENDING' in p.get_text() for p in d))"

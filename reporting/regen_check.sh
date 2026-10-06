#!/bin/bash
# Submission gate: regenerate every output of the paper from the records into a fresh copy and compare each
# byte for byte with the paper's generated/ and figures/. Prints DIFFERENT for any stale output.
#   bash reporting/regen_check.sh <paper> [--write]
# --write regenerates into the paper itself (run, review, then rerun without --write to confirm).
# Uses python3 from the active environment; set PYTHON to choose another interpreter.
set -u
cd "$(dirname "$0")/.."
if [ -d paper ]; then DEFAULT_PAPER=paper; else DEFAULT_PAPER=$HOME/workspace/papers/www2027; fi
PAPER=$(realpath "${1:-$DEFAULT_PAPER}")
PY=${PYTHON:-python3}
if [ "${2:-}" = "--write" ]; then T=$PAPER; else
  T=$(mktemp -d)/paper; mkdir -p "$T/generated/random" "$T/figures"; fi   # empty: a failed generator shows up
G=$T/generated; F=$T/figures
failed=0
run() { "$PY" "$@" > /dev/null 2>&1 || { echo "FAILED: $*"; failed=$((failed+1)); }; }
run scripts/make_dataset_table.py --paper "$T"
run analysis/chain.py confirmatory/chain --seeds 5 --out "$G"
run analysis/native.py confirmatory/native --features spectral --out "$G"
run analysis/native.py confirmatory/native --features random --out "$G/random"
run exploratory/analyze.py --out "$G"
run exploratory/posthoc_per_edge.py --out "$G"
run exploratory/posthoc_reach_max.py --collect --out "$G"
run exploratory/posthoc_reach_max.py --native --collect --out "$G"
run replication/analyze.py --out "$G"
run exploratory/e5_distance.py --collect --out "$G"
run exploratory/e6_measure.py --collect --out "$G"
run exploratory/e7_audit.py --collect --out "$G"
run exploratory/e7_audit.py --subset uniform --collect --out "$G"
run exploratory/e8_cycles.py --collect --out "$G"
run reporting/sign_mass.py --paper "$T"
run reporting/protocol_assets.py --paper "$T"
run reporting/robustness.py --paper "$T"
CUDA_VISIBLE_DEVICES=${GPU:-0} run reporting/shells.py --paper "$T" --device cuda:0
run revision/exhaustive.py --collect --out "$G"
run revision/native.py --collect --out "$G"
run revision/assets.py --paper "$T"
run reporting/paper_assets.py --paper "$T"
run collective/stage2/report.py --paper "$T"
[ "${2:-}" = "--write" ] && { echo "regenerated in place: $PAPER ($failed generator failures)"; exit $failed; }
bad=$failed
# expected outputs: every generated/ or figures/ path RESULT_MAP lists, except rows marked historical
expected=$(grep -v "historical" RESULT_MAP.md | grep -o -E '`(generated|figures)/[^`]+`' | tr -d '`' | sort -u)
present=$(cd "$PAPER" && find generated figures -type f \( -name '*.tex' -o -name '*.json' -o -name '*.pdf' \) | sort)
for f in $(comm -13 <(echo "$expected") <(echo "$present")); do echo "UNMAPPED  $f (in the paper, not in RESULT_MAP)"; bad=$((bad+1)); done
for f in $expected; do
  [ -f "$PAPER/$f" ] || { echo "ABSENT    $f (RESULT_MAP lists it, the paper lacks it)"; bad=$((bad+1)); }
done
for f in $expected; do
  if [ ! -f "$T/$f" ]; then echo "MISSING   $f (no generator produced it)"; bad=$((bad+1))
  elif ! cmp -s "$PAPER/$f" "$T/$f"; then echo "DIFFERENT $f"; bad=$((bad+1)); fi
done
echo "$bad problems: stale, missing, unmapped or failed (fresh copy: $T)"
exit $bad

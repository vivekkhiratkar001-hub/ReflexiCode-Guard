#!/usr/bin/env bash
# Final regression gate (22 Oct) - the same commands CI runs. Exit non-zero on ANY failure.
#   1. all unit + integration + E2E tests
#   2. offline end-to-end review of the sample PR
#   3. benchmark: 6 variants x N repeats, reproducibility (incl. fresh interpreter) + frozen-baseline compare
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OUT="${1:-results/regression}"

echo "== 1/3 tests ==";            $PY -m unittest discover -s tests
echo "== 2/3 offline E2E review =="; $PY -m reflexicode_guard --offline tests/fixtures --output "$OUT-offline.json" > /dev/null
echo "== 3/3 benchmark vs frozen baseline =="
$PY -m reflexicode_guard.e2e_runner --bench benchmarks/controlled_prs --out "$OUT" --repeats "${REPEATS:-5}" \
    --baseline benchmarks/baseline_summary.json
echo "REGRESSION PASSED"

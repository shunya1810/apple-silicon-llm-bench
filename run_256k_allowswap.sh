#!/bin/zsh
# Diagnostic (2026-10-13): the M1 build v2.12.2-m1.1 refused the 256K turn 1 (memory guard: 54.7 GiB
# projected vs 48.0 GiB). One run with --allow-swap to measure the real peak; swap usage logged each minute.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
caffeinate -dimsu -w $$ &
( while true; do echo "$(date +%T) $(sysctl -n vm.swapusage) | $(memory_pressure -Q 2>/dev/null | tail -1)"; sleep 60; done ) > results/allowswap256-swap.log 2>&1 &
SW=$!
"$PY" orchestrate.py --plan plans/refresh-1013-256k-allowswap.json --out results/2026-09-m1max-64gb/rows.jsonl
kill $SW
touch results/2026-09-m1max-64gb/ALLOWSWAP256_DONE

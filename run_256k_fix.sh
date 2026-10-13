#!/bin/zsh
# Admission fix check (2026-10-13): commit 61af5233, default settings (no --allow-swap), 256K once.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
caffeinate -dimsu -w $$ &
( while true; do echo "$(date +%T) $(sysctl -n vm.swapusage) | $(memory_pressure -Q 2>/dev/null | tail -1)"; sleep 60; done ) > results/fix256-swap.log 2>&1 &
SW=$!
"$PY" orchestrate.py --plan plans/refresh-1013-256k-61af5233.json --out results/2026-09-m1max-64gb/rows.jsonl
kill $SW
touch results/2026-09-m1max-64gb/FIX256_DONE

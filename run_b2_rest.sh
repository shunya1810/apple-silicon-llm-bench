#!/bin/zsh
# Resume of run_b2.sh after the 27B 246K oMLX cell (its canary-slow re-run was skipped by hand, 2026-10-19 00:12).
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-10-m1max-64gb-b2
caffeinate -i "$PY" orchestrate.py --plan plans/b2-27b-rest.json --out $OUT/rows-27b.jsonl
caffeinate -i "$PY" orchestrate.py --plan plans/b2-a3b.json --out $OUT/rows-a3b.jsonl
touch $OUT/DONE

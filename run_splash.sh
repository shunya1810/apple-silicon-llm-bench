#!/bin/zsh
# Splash M1 build, same scenarios as the other engines: 2K-64K in two rounds, then 128K once.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/splash-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/splash-night.json --out $OUT

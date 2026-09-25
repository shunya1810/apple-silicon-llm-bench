#!/bin/zsh
# 2K-64K plan (two rounds, engine order reversed in round 2)
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/day.json --out $OUT
touch results/2026-09-m1max-64gb/DAY_DONE

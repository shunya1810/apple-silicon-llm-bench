#!/bin/zsh
# 128K plan (one round, upstream -> oMLX -> fork), appended to the same results file
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/night.json --out $OUT
touch results/2026-09-m1max-64gb/NIGHT_DONE

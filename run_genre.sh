#!/bin/zsh
# Genre x length bench (code / English prose / Japanese prose / agent log, 2K-128K), fork v2.12.2-m1.10 vs Splash 1.3.2-m1.1.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-10-m1max-64gb-genre
mkdir -p $OUT
for m in a3b 27b; do
  caffeinate -i "$PY" orchestrate.py --plan plans/genre-$m.json --out $OUT/rows-$m.jsonl
done
touch $OUT/DONE

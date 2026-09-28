#!/bin/zsh
# TensorFold 0.3.5.1, same scenarios: 2K-64K in two rounds, then 128K once (bf16 KV: it has no 8-bit KV).
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/tensorfold-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/tensorfold-night.json --out $OUT
touch results/2026-09-m1max-64gb/TENSORFOLD_DONE

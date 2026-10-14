#!/bin/zsh
# A3B, thinking on, short prompts: fork (device-resident decode) vs Splash 1.3.0-m1.1, alternating.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-10-m1max-64gb-a3b-think
for e in mtplx-fork-a3b-res splash-m1-a3b mtplx-fork-a3b-res splash-m1-a3b; do
  "$PY" scripts/think_short.py --engine engines/$e.json --out $OUT --n 1 2>&1 | grep -E '^\{|Error'
  sleep 60
done
touch $OUT/DONE

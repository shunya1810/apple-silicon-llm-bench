#!/bin/zsh
# Bench 2 (2026-10-18): MTPLX M1 fork vs MTPLX upstream v2.12.2 vs oMLX 0.7.1rc1 vs Splash 1.3.2-m1.1,
# thinking on, 2K-246K, Qwen3.8-27B then Qwen3.6-35B-A3B, same body weights for every engine.
# Each cell waits for >=30% battery on AC (orchestrate.py), and the guard in bench-engines/guard stops GPU work at 15%.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-10-m1max-64gb-b2
mkdir -p $OUT
for m in ${=B2_MODELS:-27b a3b}; do
  caffeinate -i "$PY" orchestrate.py --plan plans/b2-$m.json --out $OUT/rows-$m.jsonl
done
touch $OUT/DONE

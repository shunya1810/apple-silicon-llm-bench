#!/bin/zsh
# Re-runs (2026-09-26): Splash 128K (attempt 3), then the fork at 40b6113 (context-copy off on M1).
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" runner.py --engine engines/splash-m1.json --scenario scenarios/mt-128k.json --out $OUT --round 1 --order-pos 1 --kv q8 --attempt 3
sleep 120
"$PY" orchestrate.py --plan plans/fork-40b6113-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/fork-40b6113-night.json --out $OUT

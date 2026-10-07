#!/bin/zsh
# oMLX 0.7.0 re-run (2026-10-07): the same cells as the 0.6.4 run (2K fp16/q8,
# 8K-64K q8, two rounds; 128K one run) plus the 256K spot check, which 0.6.4
# never ran. The 0.6.4 rows stay in rows.jsonl as omlx-064. A fork 2K probe
# (canary 31.1 tok/s, gate >= 29.9) confirmed the machine state first.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
"$PY" orchestrate.py --plan plans/omlx070-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/omlx070-night.json --out $OUT
touch results/2026-09-m1max-64gb/OMLX070_DONE
sleep 300
until [ "$(battery)" -ge 80 ]; do sleep 60; done
echo "$(date +%T) start 256K battery $(battery)% $(pmset -g ac | grep Wattage)"
"$PY" orchestrate.py --plan plans/spot-256k-omlx.json --out $OUT
echo "$(date +%T) end 256K battery $(battery)%"
touch results/2026-09-m1max-64gb/OMLX070_256_DONE

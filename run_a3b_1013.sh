#!/bin/zsh
# Qwen3.6-35B-A3B (MoE): MTPLX M1 fork v2.12.2-m1.2 (MTP depth 1) against Splash 1.3.0-m1.1,
# interleaved, 2K-64K two rounds and 128K once. Earlier A3B rows (fork 0cd2a73, qwisp) stay.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb-qwen36-a3b/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
wait_battery() { until [ "$(battery)" -ge 80 ]; do sleep 60; done; echo "$(date +%T) battery $(battery)% $(pmset -g ac | grep Wattage)"; }
wait_battery; "$PY" orchestrate.py --plan plans/a3b-1013-day.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/a3b-1013-night.json --out $OUT
touch results/2026-09-m1max-64gb-qwen36-a3b/A3B1013_DONE

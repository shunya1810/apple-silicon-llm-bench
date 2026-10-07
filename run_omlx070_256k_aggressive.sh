#!/bin/zsh
# 256K for oMLX 0.7.0 with memory_guard_tier=aggressive (2026-10-07). At its default
# (balanced) the prefill memory guard rejected the 257,672-token first turn with HTTP 400
# (~46.1 GB predicted peak against a 39.4 GB dynamic ceiling); that row stays in rows.jsonl
# under omlx. This is the only non-default oMLX setting in the benchmark.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
until [ "$(battery)" -ge 80 ]; do sleep 60; done
echo "$(date +%T) start 256K aggressive battery $(battery)% $(pmset -g ac | grep Wattage)"
"$PY" orchestrate.py --plan plans/spot-256k-omlx-aggressive.json --out $OUT
echo "$(date +%T) end 256K aggressive battery $(battery)%"
touch results/2026-09-m1max-64gb/OMLX070_256_AGGR_DONE

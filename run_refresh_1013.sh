#!/bin/zsh
# Re-run (2026-10-13) of the two leading engines at their latest releases: the MTPLX
# M1 build at v2.12.2-m1.1 and Splash 1.3.0-m1.1, interleaved,
# with the same plans: 2K-64K two rounds (order reversed), 128K once, the sampled
# scenario, then 256K once each (fork first). Earlier rows stay as mtplx-fork-m1-2122 and splash-m1-121.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
wait_battery() { until [ "$(battery)" -ge 80 ]; do sleep 60; done; echo "$(date +%T) battery $(battery)% $(pmset -g ac | grep Wattage)"; }
wait_battery; "$PY" orchestrate.py --plan plans/refresh-1013-day.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/refresh-1013-night.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/refresh-1013-sampling.json --out results/2026-09-m1max-64gb-sampling/rows.jsonl
touch results/2026-09-m1max-64gb/REFRESH1013_DONE
for e in mtplx-fork splash-m1; do
  sleep 300; wait_battery
  echo "$(date +%T) start 256K $e"
  "$PY" orchestrate.py --plan plans/refresh-1013-256k-$e.json --out $OUT
  echo "$(date +%T) end 256K $e"
done
touch results/2026-09-m1max-64gb/REFRESH1013_256_DONE

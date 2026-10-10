#!/bin/zsh
# Re-run (2026-10-08) of the two leading engines at their latest releases: the MTPLX
# M1 build at v2.12.2-m1 (upstream 2.12.2 base) and Splash 1.2.1-m1, interleaved,
# with the same plans: 2K-64K two rounds (order reversed), 128K once, the sampled
# scenario, then 256K once each (Splash first). Earlier rows stay as mtplx-fork-m1.2
# and splash-m1-110. 2K smoke runs of both engines first read canaries of 30.7 (fork)
# and 28.9 tok/s (Splash).
# The fork's 256K plan did not run: v2.12.2-m1 had refused turn 2 at 128K for memory, so the
# script was stopped by hand once Splash's 256K run had started.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
wait_battery() { until [ "$(battery)" -ge 80 ]; do sleep 60; done; echo "$(date +%T) battery $(battery)% $(pmset -g ac | grep Wattage)"; }
wait_battery; "$PY" orchestrate.py --plan plans/refresh-1008-day.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/refresh-1008-night.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/refresh-1008-sampling.json --out results/2026-09-m1max-64gb-sampling/rows.jsonl
touch results/2026-09-m1max-64gb/REFRESH1008_DONE
for e in splash-m1 mtplx-fork; do
  sleep 300; wait_battery
  echo "$(date +%T) start 256K $e"
  "$PY" orchestrate.py --plan plans/refresh-1008-256k-$e.json --out $OUT
  echo "$(date +%T) end 256K $e"
done
touch results/2026-09-m1max-64gb/REFRESH1008_256_DONE

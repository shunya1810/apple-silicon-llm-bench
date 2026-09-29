#!/bin/zsh
# 256K spot check (2026-09-29): the two engines that lead at 128K, one run each,
# Splash first so the M1 build runs on the warmer machine. Each waits for the
# battery to reach 80% first: on the 100 W adapter a long GPU run drains it and
# the machine slows down once it runs low.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
for e in splash-m1 mtplx-fork-m1.1; do
  until [ "$(battery)" -ge 80 ]; do sleep 60; done
  echo "$(date +%T) start $e battery $(battery)% $(pmset -g ac | grep Wattage)"
  "$PY" orchestrate.py --plan plans/spot-256k-$e.json --out $OUT
  echo "$(date +%T) end $e battery $(battery)%"
done
touch results/2026-09-m1max-64gb/SPOT256_DONE

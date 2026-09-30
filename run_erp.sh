#!/bin/zsh
# Short-prompt head-to-head in the style of "Splash on M1, part 2" (benchmark 5): the MTPLX
# M1 build v2.12.0-m1.2 and Splash 1.1.0-m1, ABBA, each run = npanj's five prompts on a
# 5-minute loop (250 tokens, reasoning xhigh) + rest + one cold 44K prompt, with macmon.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb-erp
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
r=0
for e in fork splash splash fork; do
  r=$((r + 1))
  until [ "$(battery)" -ge 90 ]; do sleep 60; done
  echo "$(date +%T) run $r $e battery $(battery)% $(pmset -g ac | grep Wattage)"
  "$PY" scripts/erp_repro.py --engine engines/erp/$e.json --out $OUT --round $r
  sleep 300
done
touch $OUT/ERP_DONE

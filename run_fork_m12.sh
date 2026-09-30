#!/bin/zsh
# Re-run (2026-09-30): the MTPLX M1 build at v2.12.0-m1.2 (command buffers off by default,
# SSD cache fix), alone, with the same plans. The v2.12.0-m1 rows stay as mtplx-fork-m1.
# Each plan waits for the battery to reach 90% (100 W adapter).
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
wait_battery() { until [ "$(battery)" -ge 90 ]; do sleep 60; done; echo "$(date +%T) battery $(battery)% $(pmset -g ac | grep Wattage)"; }
# Probe: one 2K cell to a scratch file; go only when the canary decodes at >= 29.9 tok/s
# (m1.2 on a cool, idle machine: 30.3-30.5). Afternoons have run ~5% slower.
PROBE=results/2026-09-m1max-64gb/probe-fork-m12.jsonl
while true; do
  wait_battery
  "$PY" runner.py --engine engines/mtplx-fork.json --scenario scenarios/mt-2k.json --out $PROBE > /dev/null 2>&1
  c=$(python3 -c "import json;print([json.loads(l) for l in open('$PROBE')][-4]['decode_tok_s'])")
  echo "$(date +%T) probe canary $c"
  python3 -c "import sys;sys.exit(0 if float('$c')>=29.9 else 1)" && break
  [ "$(date +%H)" -ge 5 ] && [ "$(date +%H)" -lt 12 ] && { echo "giving up"; exit 1; }
  sleep 1800
done
sleep 120
wait_battery; "$PY" orchestrate.py --plan plans/fork-m12-day.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/fork-m12-night.json --out $OUT
sleep 300; wait_battery; "$PY" orchestrate.py --plan plans/fork-m12-sampling.json --out results/2026-09-m1max-64gb-sampling/rows.jsonl
touch results/2026-09-m1max-64gb/FORK_M12_DONE

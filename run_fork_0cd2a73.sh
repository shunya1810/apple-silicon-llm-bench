#!/bin/zsh
# Re-run (2026-09-27): the fork at 0cd2a73 (M1: larger Metal command buffers,
# prefill evaluated every 4 layers to keep the peak). The bba1b7b rows stay in
# rows.jsonl as mtplx-fork-bba1b7b.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/fork-0cd2a73-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/fork-0cd2a73-night.json --out $OUT
touch results/2026-09-m1max-64gb/FORK_0CD2A73_DONE

#!/bin/zsh
# Re-run (2026-09-26): the fork at 4fe8067 (M1 memory defaults: MLX cache 1 GiB,
# 2 bank entries per session, MMA prefill from the first chunk). The 40b6113 rows
# stay in rows.jsonl as mtplx-fork-40b6113.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/fork-4fe8067-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/fork-4fe8067-night.json --out $OUT
touch results/2026-09-m1max-64gb/FORK_4FE8067_DONE

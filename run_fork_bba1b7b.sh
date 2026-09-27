#!/bin/zsh
# Re-run (2026-09-27): the fork at bba1b7b (M1: gdn/attn projection fusion and the
# FR-Spec draft head on the live draft path, Japanese-aware list). The 4fe8067 rows
# stay in rows.jsonl as mtplx-fork-4fe8067.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/fork-bba1b7b-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/fork-bba1b7b-night.json --out $OUT
touch results/2026-09-m1max-64gb/FORK_BBA1B7B_DONE

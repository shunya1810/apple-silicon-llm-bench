#!/bin/zsh
# Fork-only re-run of 32K/64K (2 rounds) and 128K after the mfa_attn metallib was missing from the worktree.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb-qwen36-a3b/rows.jsonl
until [ -f results/2026-09-m1max-64gb-qwen36-a3b/A3B1014_DONE ]; do sleep 30; done
caffeinate -dimsu -w $$ &
sleep 120; "$PY" orchestrate.py --plan plans/a3b-1014-fork-long.json --out $OUT
sleep 120; "$PY" orchestrate.py --plan plans/a3b-1014-fork-128k.json --out $OUT
touch results/2026-09-m1max-64gb-qwen36-a3b/A3B1014_FORK_DONE

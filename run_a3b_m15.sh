#!/bin/zsh
# fork v2.12.2-m1.5 against Splash 1.3.0-m1.1 on Qwen3.6-35B-A3B: 2K-64K two rounds (interleaved), 128K once.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb-qwen36-a3b/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/a3b-1014-m15.json --out $OUT
sleep 120; "$PY" orchestrate.py --plan plans/a3b-1014-m15-128k.json --out $OUT
touch results/2026-09-m1max-64gb-qwen36-a3b/A3B_M15_DONE

#!/bin/zsh
# Same-session re-run (2026-09-29) for the public comparison: the M1 build at
# its release tag, upstream 1de2b1c at defaults, and upstream with every M1-build
# default that upstream exposes as an environment variable (engines/
# mtplx-upstream-tuned.json), interleaved. Then a sampled scenario (temperature
# 1.0 / top-p 0.95 / top-k 20, 512 tokens) for the three plus Splash 1.1.0-m1.
# Earlier rows stay in rows.jsonl as mtplx-fork-0cd2a73 and mtplx-upstream-0925.
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
OUT=results/2026-09-m1max-64gb/rows.jsonl
caffeinate -dimsu -w $$ &
"$PY" orchestrate.py --plan plans/fair-day.json --out $OUT
sleep 300
"$PY" orchestrate.py --plan plans/fair-night.json --out $OUT
touch results/2026-09-m1max-64gb/FAIR_GREEDY_DONE
sleep 300
"$PY" orchestrate.py --plan plans/sampling.json --out results/2026-09-m1max-64gb-sampling/rows.jsonl
touch results/2026-09-m1max-64gb/FAIR_DONE

# Follow-ups that ran after the above (2026-09-29): upstream + env without the
# MLX command-buffer variables (it raised memory and ran out at 128K with them).
"$PY" orchestrate.py --plan plans/fair-nobuf.json --out $OUT
"$PY" orchestrate.py --plan plans/fair-nobuf-night.json --out $OUT
"$PY" orchestrate.py --plan plans/fair-nobuf-fill.json --out $OUT
"$PY" orchestrate.py --plan plans/sampling-nobuf.json --out results/2026-09-m1max-64gb-sampling/rows.jsonl

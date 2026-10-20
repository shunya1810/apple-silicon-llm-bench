#!/usr/bin/env python3
"""Genre x length table: fork vs Splash per model (results/2026-10-m1max-64gb-genre).

  python scripts/summarize_genre.py [run dir]

Per cell: prompt tokens, prefill tok/s (prompt / time to first token), decode tok/s, tokens per verify round
(fork: completion / verify calls; Splash: decoded tokens / decode batches from its /status counters), peak wired.
"""
import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026-10-m1max-64gb-genre")
GENRES = [("code", "code (CPython stdlib)"), ("en", "English novel"), ("ja", "Japanese novel"), ("agent", "agent log")]
LENS = ["2k", "8k", "32k", "128k"]


def per_round(r):
    s = r.get("engine_stats") or {}
    if "verify_calls" in s and s["verify_calls"]:
        return r["completion_tokens"] / s["verify_calls"]
    batches = s.get("decode_batches")
    if batches:
        return r["completion_tokens"] / batches
    acc = s.get("draft_acceptance_rate")
    return None if acc is None else None


def load(model):
    cells = {}
    p = RUN / f"rows-{model}.jsonl"
    if not p.exists():
        return cells
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") != "turn":
            continue
        eng = "fork" if "-fork-" in r["engine"] else "splash"
        cells[(eng, r["scenario"])] = r      # the last attempt wins
    return cells


def cell(r, key):
    if r is None:
        return "-"
    if r.get("error"):
        return "error"
    if key == "prefill":
        return f"{(r['prompt_tokens'] or 0) / r['ttft_s']:,.0f}" if r.get("ttft_s") else "-"
    if key == "decode":
        return f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") else "-"
    if key == "round":
        v = per_round(r)
        return "-" if v is None else f"{v:.2f}"
    if key == "wired":
        return f"{(r.get('peak_wired') or 0) / 1e9:.1f}"
    return "-"


for model in ("a3b", "27b"):
    cells = load(model)
    if not cells:
        continue
    print(f"\n## {model}\n")
    for key, title in (("prefill", "prefill tok/s"), ("decode", "decode tok/s"), ("round", "tokens per verify round"),
                       ("wired", "peak wired GB")):
        print(f"**{title}** (fork / Splash)\n")
        print("| genre | " + " | ".join(l.upper() for l in LENS) + " |")
        print("|---|" + "---|" * len(LENS))
        for g, name in GENRES:
            vals = []
            for lab in LENS:
                sc = f"g-{g}-{lab}"
                vals.append(f"{cell(cells.get(('fork', sc)), key)} / {cell(cells.get(('splash', sc)), key)}")
            print(f"| {name} | " + " | ".join(vals) + " |")
        print()

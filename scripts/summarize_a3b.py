#!/usr/bin/env python3
"""Tables (en/ja) for the Qwen3.6-35B-A3B (MoE) runs.

  python scripts/summarize_a3b.py results/2026-09-m1max-64gb-qwen36-a3b

Per (engine, scenario, turn) the value is the median over rounds of the last
attempt of each round; rows with an error are left out. Each engine's rows
come from its run in RUN (a run_stamp prefix).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ENGINES = ["mtplx-fork-a3b-moe", "splash-m1-a3b", "mtplx-fork-a3b-m12"]
NAMES = {
    "mtplx-fork-a3b-moe": "MTPLX M1 fork (m1.4)",
    "splash-m1-a3b": "Splash 1.3.0-m1.1",
    "mtplx-fork-a3b-m12": "MTPLX M1 fork (m1.2)",
}
# The run each engine's rows come from (run_stamp prefix): the m1.4 / Splash pair
# was measured together on 2026-10-14, m1.2 the day before.
RUN = {"mtplx-fork-a3b-moe": "20261014", "splash-m1-a3b": "20261014", "mtplx-fork-a3b-m12": "20261013"}
CTX = ["mt-2k", "mt-8k", "mt-32k", "mt-64k", "mt-128k", "mt-256k"]
LABEL = {"mt-2k": "2K", "mt-8k": "8K", "mt-32k": "32K", "mt-64k": "64K", "mt-128k": "128K", "mt-256k": "256K"}


def load(run: Path, prefix: str):
    last: dict[tuple, dict] = {}
    for line in (run / "rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") != "turn" or r.get("error") or r["engine"] not in NAMES:
            continue
        if not str(r.get("run_stamp", "")).startswith(prefix or RUN[r["engine"]]):
            continue
        key = (r["engine"], r["scenario"], r.get("round"), r.get("turn"))
        if key not in last or int(r.get("attempt") or 1) >= int(last[key].get("attempt") or 1):
            last[key] = r
    cells: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for (e, s, _rnd, t), r in last.items():
        cells[(e, s)][int(t)].append(r)
    return cells


def med(rows, field):
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return statistics.median(vals) if vals else None


def summary(cells, e, s):
    turns = cells.get((e, s))
    if not turns or set(turns) != {1, 2, 3}:
        return None
    conv = sum(med(turns[t], "e2e_s") for t in (1, 2, 3))
    return {
        "conv": conv,
        "ttft1": med(turns[1], "ttft_s"),
        "ttft23": statistics.mean([med(turns[t], "ttft_s") for t in (2, 3)]),
        "decode": statistics.mean([med(turns[t], "decode_tok_s") for t in (1, 2, 3)]),
        "prefill": med(turns[1], "prompt_tokens") / med(turns[1], "ttft_s"),
        "peak": max(float(r.get("peak_wired") or 0) for t in turns for r in turns[t]) / 1e9,
        "rounds": min(len(turns[t]) for t in turns),
    }


def dur(sec):
    return f"{sec:.1f} s" if sec < 120 else f"{sec / 60:.1f} min"


def tables(cells, lang):
    head = {
        "en": "| Context | Engine | Whole conversation | Turn 1 TTFT | Turns 2–3 TTFT | Cold prefill | Decode (mean of 3 turns) | Peak wired | Rounds |",
        "ja": "| 文脈 | エンジン | 会話全体 | 1ターン目の応答開始 | 2・3ターン目の応答開始 | cold の prefill | decode（3ターンの平均） | ピークのメモリ | 回数 |",
    }[lang]
    out = [head, "|" + "---|" * 9]
    for s in CTX:
        for e in ENGINES:
            v = summary(cells, e, s)
            if v is None:
                continue
            bold = "**" if e == ENGINES[0] else ""
            out.append(
                f"| {LABEL[s]} | {bold}{NAMES[e]}{bold} | {bold}{dur(v['conv'])}{bold} | {dur(v['ttft1'])} | "
                f"{v['ttft23']:.1f} s | {v['prefill']:.0f} tok/s | {bold}{v['decode']:.1f} tok/s{bold} | "
                f"{v['peak']:.1f} GB | {v['rounds']} |"
            )
    return "\n".join(out)


def main():
    run = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    cells = load(run, prefix)
    for lang in ("en", "ja"):
        text = tables(cells, lang)
        (run / f"tables.{lang}.md").write_text(text + "\n")
        print(text)
        print()


if __name__ == "__main__":
    main()

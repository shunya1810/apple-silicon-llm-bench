#!/usr/bin/env python3
"""Tables for the sampled scenarios (sm-*: temperature 1.0, top-p 0.95, top-k 20, 512 tokens).

  python scripts/summarize_sampling.py results/<run>-sampling

Same aggregation as summarize.py (median over rounds of the last attempt per round);
writes summary.json and tables.{en,ja}.md next to rows.jsonl. No charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ENGINES, NAMES, build, f, fs, load, mean_turns, turn  # noqa: E402

CTX = ["sm-2k", "sm-32k", "sm-64k"]
CTX_LABEL = {"sm-2k": "2K", "sm-32k": "32K", "sm-64k": "64K"}


def tables_md(cells: dict, lang: str) -> str:
    ja = lang == "ja"
    head = ("| context | engine | T1 TTFT (cold) | decode T1 / T2 / T3 (tok/s) | decode mean | "
            "draft acceptance T1 / T2 / T3 | tokens T1 / T2 / T3 | needle |") if not ja else (
            "| コンテキスト | エンジン | T1 TTFT（コールド） | decode T1 / T2 / T3 (tok/s) | decode 平均 | "
            "ドラフト受理率 T1 / T2 / T3 | 生成トークン T1 / T2 / T3 | needle |")
    out = [head, "|---|---|---|---|---|---|---|---|"]
    for s in CTX:
        for e in ENGINES:
            if (e, "q8", s) not in cells:
                continue
            g = lambda t, k: turn(cells, e, s, t, k)
            acc = [g(t, "accept_rate") for t in (1, 2, 3)]
            acc_s = "—" if all(a is None for a in acc) else " / ".join("—" if a is None else f"{a:.0%}" for a in acc)
            needles = cells[(e, "q8", s)]["turns"].get(3, {}).get("needle_ok", [])
            nd = "—" if not needles else f"{sum(needles)}/{len(needles)}"
            out.append(
                f"| {CTX_LABEL[s]} | {NAMES[e]} | {fs(g(1, 'ttft_s'))} | "
                f"{f(g(1, 'decode_tok_s'))} / {f(g(2, 'decode_tok_s'))} / {f(g(3, 'decode_tok_s'))} | "
                f"**{f(mean_turns(cells, e, s, 'decode_tok_s'))}** | {acc_s} | "
                f"{f(g(1, 'completion_tokens'), 0)} / {f(g(2, 'completion_tokens'), 0)} / "
                f"{f(g(3, 'completion_tokens'), 0)} | {nd} |")
    return "\n".join(out) + "\n"


def main() -> None:
    run = Path(sys.argv[1])
    cells = build(load(run))
    (run / "summary.json").write_text(json.dumps({"|".join(k): v for k, v in cells.items()}, indent=1, default=list))
    for lang in ("en", "ja"):
        (run / f"tables.{lang}.md").write_text(tables_md(cells, lang))
    print("cells:", len(cells))


if __name__ == "__main__":
    main()

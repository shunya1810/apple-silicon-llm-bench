#!/usr/bin/env python3
"""Tables for bench 2 (thinking on, 2K-246K, five engines, two models).

  python scripts/summarize_b2.py [results/2026-10-m1max-64gb-b2]

Per cell the last attempt counts. Columns: cold prefill (turn 1 time to first token and
prompt tokens / that time), decode (completion tokens over streaming time, mean of the
three turns, thinking tokens included), turns 2-3 time to first token, peak wired memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "results/2026-10-m1max-64gb-b2")
# The whole-conversation time is left out of the README tables: at temperature 1.0 each
# engine thinks for a different number of tokens, so it would compare different work.
ORDER = ["fork", "up", "omlx", "splash", "omlx-oq4e"]
NAMES = {"fork": "MTPLX M1 fork m1.9", "up": "MTPLX upstream 2.12.2", "omlx": "oMLX 0.7.1rc1",
         "splash": "Splash 1.3.2-m1.1", "omlx-oq4e": "oMLX 0.7.1rc1 + Jundot oQ4e"}
CTX = ["th-2k", "th-8k", "th-32k", "th-64k", "th-128k", "th-256k"]


def rows(model: str) -> dict:
    path = RUN / f"rows-{model}.jsonl"
    cells: dict = {}
    if not path.exists():
        return cells
    for line in path.read_text().splitlines():
        r = json.loads(line)
        key = (r["engine"], r["scenario"])
        att = r.get("attempt", 1)
        cur = cells.get(key)
        if cur is None or att > cur["attempt"]:
            cells[key] = {"attempt": att, "rows": []}
        if att == cells[key]["attempt"]:
            cells[key]["rows"].append(r)
    return cells


def fmt(v, nd=1):
    return "-" if v is None else f"{v:,.{nd}f}"


def table(model: str) -> str:
    cells = rows(model)
    out = [f"## {model}", "",
           "| context | engine | prompt | cold TTFT s | prefill tok/s | decode tok/s (t1/t2/t3) | TTFT t2/t3 s | peak wired GB | finish | needle |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for ctx in CTX:
        for e in ORDER:
            c = cells.get((f"b2-{e}-{model}", ctx))
            if not c:
                continue
            turns = {r["turn"]: r for r in c["rows"] if r["phase"] == "turn"}
            t1 = turns.get(1)
            if not t1:
                err = next((r.get("error") for r in c["rows"] if r.get("error")), "no turn")
                out.append(f"| {ctx} | {NAMES[e]} | - | - | - | - | - | - | {str(err)[:60]} | - |")
                continue
            pre = (t1["prompt_tokens"] / t1["ttft_s"]) if t1.get("prompt_tokens") and t1.get("ttft_s") else None
            dec = [turns[i].get("decode_tok_s") if i in turns else None for i in (1, 2, 3)]
            ttft23 = "/".join(fmt(turns[i]["ttft_s"]) if i in turns and turns[i].get("ttft_s") else "-" for i in (2, 3))
            peak = max((r.get("peak_wired") or 0) for r in turns.values()) / 1e9
            fin = "/".join((turns[i].get("finish_reason") or "err")[:4] if i in turns else "-" for i in (1, 2, 3))
            needle = turns.get(3, {}).get("needle_ok")
            out.append(f"| {ctx} | {NAMES[e]} | {t1.get('prompt_tokens') or '-'} | {fmt(t1.get('ttft_s'))} | {fmt(pre, 0)} | "
                       f"{'/'.join(fmt(d) for d in dec)} | {ttft23} | {fmt(peak)} | {fin} | {needle} |")
    return "\n".join(out)


def main_tables() -> None:
    print(table("27b"))
    print()
    print(table("a3b"))


# ---------------------------------------------------------------- README tables

def metric_tables(model: str, lang: str = "en") -> str:
    """Engines as columns, contexts as rows: cold prefill, decode, follow-up TTFT, whole conversation, peak."""
    cells = rows(model)
    engines = [e for e in ORDER if any(k[0] == f"b2-{e}-{model}" for k in cells)]
    ja = lang == "ja"
    head = ("| 文脈 | " if ja else "| context | ") + " | ".join(NAMES[e] for e in engines) + " |"
    fail = "失敗" if ja else "failed"
    sep = "|---|" + "---|" * len(engines)
    label = {"2k": "2K", "8k": "8K", "32k": "32K", "64k": "64K", "128k": "128K", "256k": "246K"}

    def cell(e, ctx):
        c = cells.get((f"b2-{e}-{model}", ctx))
        if not c:
            return None
        return {r["turn"]: r for r in c["rows"] if r["phase"] == "turn"}, c["rows"]

    def render(title, fn):
        out = [title, "", head, sep]
        for ctx in CTX:
            vals = []
            for e in engines:
                got = cell(e, ctx)
                vals.append("-" if got is None else fn(*got))
            out.append(f"| {label[ctx.split('-')[1]]} | " + " | ".join(vals) + " |")
        return "\n".join(out)

    def failed(turns, all_rows):
        t1 = turns.get(1)
        return t1 is None or t1.get("error") or not t1.get("ttft_s")

    def prefill(turns, all_rows):
        if failed(turns, all_rows):
            return fail
        t1 = turns[1]
        unit = "秒" if ja else " s"
        return f"{t1['ttft_s']:,.0f}{unit} ({t1['prompt_tokens'] / t1['ttft_s']:,.0f} tok/s)"

    def decode(turns, all_rows):
        if failed(turns, all_rows):
            return fail
        return " / ".join(f"{turns[i]['decode_tok_s']:.1f}" if i in turns and turns[i].get("decode_tok_s") else "-"
                          for i in (1, 2, 3))

    def follow(turns, all_rows):
        if failed(turns, all_rows):
            return fail
        return " / ".join(f"{turns[i]['ttft_s']:,.1f}" if i in turns and turns[i].get("ttft_s") else "-" for i in (2, 3))

    def whole(turns, all_rows):
        if failed(turns, all_rows) or len(turns) < 3:
            return "failed" if failed(turns, all_rows) else "-"
        return f"{sum(t['e2e_s'] for t in turns.values()) / 60:.1f} min"

    def peak(turns, all_rows):
        p = max([(r.get("peak_wired") or 0) for r in all_rows if r["phase"] == "turn"] or [0]) / 1e9
        if not p:
            return "-"
        return f"{p:.1f} GB" + (f" ({fail})" if failed(turns, all_rows) else "")

    titles = (["**1 ターン目、冷えたプロンプト: 応答開始までの時間（prefill の速さ）**",
               "**decode、tok/s（1 / 2 / 3 ターン目、思考のトークンを含む）**",
               "**2 / 3 ターン目: 応答開始までの時間、秒**",
               "**wired メモリのピーク**"] if ja else
              ["**Turn 1, cold prompt: time to first token (prefill speed)**",
               "**Decode, tok/s (turns 1 / 2 / 3, thinking tokens included)**",
               "**Turns 2 / 3: time to first token, s**",
               "**Peak wired memory**"])
    return "\n\n".join(render(t, f) for t, f in zip(titles, (prefill, decode, follow, peak)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--readme":
        print(metric_tables(sys.argv[3] if len(sys.argv) > 3 else "27b",
                            sys.argv[4] if len(sys.argv) > 4 else "en"))
    else:
        main_tables()

#!/usr/bin/env python3
"""Aggregate rows.jsonl into summary tables (en/ja), a CSV and SVG charts (light/dark).

  python scripts/summarize.py results/<run>

Per (engine, kv, scenario, turn) the value is the median over rounds, using the
last attempt of each round (a cell is re-run once when its thermal canary was slow).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ENGINES = ["mtplx-fork", "mtplx-upstream", "omlx"]  # fixed order = fixed color slot
NAMES = {"mtplx-fork": "MTPLX fork", "mtplx-upstream": "MTPLX upstream", "omlx": "oMLX"}
CTX = ["mt-2k", "mt-8k", "mt-32k", "mt-64k", "mt-128k"]
CTX_LABEL = {"mt-2k": "2K", "mt-8k": "8K", "mt-32k": "32K", "mt-64k": "64K", "mt-128k": "128K"}
GB = 1e9

THEMES = {
    "light": {"bg": "#fcfcfb", "t1": "#0b0b0b", "t2": "#52514e", "grid": "#e4e3de", "axis": "#8a8983",
              "mtplx-fork": "#2a78d6", "mtplx-upstream": "#eb6834", "omlx": "#1baf7a"},
    "dark": {"bg": "#1a1a19", "t1": "#ffffff", "t2": "#c3c2b7", "grid": "#34332f", "axis": "#6d6c66",
             "mtplx-fork": "#3987e5", "mtplx-upstream": "#d95926", "omlx": "#199e70"},
}
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
MARKER = {"mtplx-fork": "dot", "mtplx-upstream": "diamond", "omlx": "square"}


# ---------------------------------------------------------------- data


def load(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "rows.jsonl").read_text().splitlines() if l.strip()]
    last: dict[tuple, int] = {}
    for r in rows:
        k = (r["engine"], r["kv"], r["scenario"], r["round"])
        last[k] = max(last.get(k, 0), r.get("attempt", 1))
    return [r for r in rows if r.get("attempt", 1) == last[(r["engine"], r["kv"], r["scenario"], r["round"])]]


def med(values):
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def build(rows: list[dict]) -> dict:
    """cells[(engine, kv, scenario)] = per-turn medians + cell-level facts."""
    cells: dict[tuple, dict] = {}
    for r in rows:
        if r["phase"] != "turn":
            continue
        c = cells.setdefault((r["engine"], r["kv"], r["scenario"]), {"turns": {}, "rounds": set(), "meta": r})
        c["rounds"].add(r["round"])
        c["turns"].setdefault(r["turn"], []).append(r)
    out = {}
    for key, c in cells.items():
        turns = {}
        for t, rs in sorted(c["turns"].items()):
            gen = [(x["e2e_s"] - x["ttft_s"]) if x.get("ttft_s") is not None else None for x in rs]
            acc = []
            for x in rs:
                s = x.get("engine_stats") or {}
                if s.get("drafted_tokens"):
                    acc.append(s["accepted_drafts"] / s["drafted_tokens"])
            turns[t] = {
                "n": len(rs),
                "prompt_tokens": med(x.get("prompt_tokens") for x in rs),
                "cached_tokens": med(x.get("cached_tokens_usage") for x in rs),
                "ttft_s": med(x.get("ttft_s") for x in rs),
                "gen_s": med(gen),
                "e2e_s": med(x.get("e2e_s") for x in rs),
                "decode_tok_s": med(x.get("decode_tok_s") for x in rs),
                "completion_tokens": med(x.get("completion_tokens") for x in rs),
                "peak_wired": max((x.get("peak_wired") or 0) for x in rs) or None,
                "peak_rss": max((x.get("peak_rss") or 0) for x in rs) or None,
                "needle_ok": [x.get("needle_ok") for x in rs if x.get("needle_ok") is not None],
                "errors": [x["error"] for x in rs if x.get("error")],
                "accept_rate": med(acc),
            }
        m = c["meta"]
        out[key] = {"engine": key[0], "kv": key[1], "scenario": key[2], "rounds": sorted(c["rounds"]),
                    "turns": turns, "kv_impl": m.get("kv_impl"), "mtp_impl": m.get("mtp_impl"),
                    "engine_version": m.get("engine_version"), "context_tokens_raw": m.get("context_tokens_raw")}
    return out


def turn(cells, engine, scen, t, field, kv="q8"):
    c = cells.get((engine, kv, scen))
    if not c or t not in c["turns"]:
        return None
    return c["turns"][t][field]


def mean_turns(cells, engine, scen, field, ts=(1, 2, 3), kv="q8"):
    v = [turn(cells, engine, scen, t, field, kv) for t in ts]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def conv_total(cells, engine, scen, kv="q8"):
    v = [turn(cells, engine, scen, t, "e2e_s", kv) for t in (1, 2, 3)]
    return sum(v) if all(x is not None for x in v) else None


# ---------------------------------------------------------------- tables


def f(v, nd=1, suffix=""):
    return "—" if v is None else f"{v:,.{nd}f}{suffix}"


def fs(v):
    if v is None:
        return "—"
    if v < 60:
        return f"{v:.1f} s" if v >= 10 else f"{v:.2f} s"
    return f"{v / 60:.1f} min"


def fgb(v):
    return "—" if v is None else f"{v / GB:.1f} GB"


def tables_md(cells: dict, lang: str) -> str:
    ja = lang == "ja"
    ctxs = [s for s in CTX if any((e, "q8", s) in cells for e in ENGINES)]
    out = []
    h = ("| context | engine | T1 TTFT (cold) | T2 TTFT | T3 TTFT | decode T1 / T2 / T3 (tok/s) | "
         "3-turn total | cached T2 / T3 | peak wired | needle |") if not ja else (
         "| 文脈 | エンジン | T1 TTFT（cold） | T2 TTFT | T3 TTFT | decode T1 / T2 / T3（tok/s） | "
         "3ターンの合計 | cache T2 / T3 | 最大 wired | needle |")
    out += [h, "|" + "---|" * 10]
    for s in ctxs:
        for e in ENGINES:
            if (e, "q8", s) not in cells:
                continue
            g = lambda t, k: turn(cells, e, s, t, k)
            needles = cells[(e, "q8", s)]["turns"].get(3, {}).get("needle_ok", [])
            nd = "—" if not needles else f"{sum(needles)}/{len(needles)}"
            wired = max((g(t, "peak_wired") or 0) for t in (1, 2, 3)) or None
            out.append(
                f"| {CTX_LABEL[s]} | {NAMES[e]} | {fs(g(1, 'ttft_s'))} | {fs(g(2, 'ttft_s'))} | {fs(g(3, 'ttft_s'))} | "
                f"{f(g(1, 'decode_tok_s'))} / {f(g(2, 'decode_tok_s'))} / {f(g(3, 'decode_tok_s'))} | "
                f"{fs(conv_total(cells, e, s))} | {f(g(2, 'cached_tokens'), 0)} / {f(g(3, 'cached_tokens'), 0)} | "
                f"{fgb(wired)} | {nd} |")
    # 2K fp16 vs q8
    if any((e, "fp16", "mt-2k") in cells for e in ENGINES):
        out += ["", "**2K: fp16 KV vs 8-bit KV** (decode tok/s, mean of turns 1–3)" if not ja else
                "**2K：fp16 KV と 8-bit KV**（decode tok/s、ターン1〜3の平均）", "",
                "| engine | fp16 KV | 8-bit KV | change |" if not ja else "| エンジン | fp16 KV | 8-bit KV | 変化 |",
                "|---|---|---|---|"]
        for e in ENGINES:
            a = mean_turns(cells, e, "mt-2k", "decode_tok_s", kv="fp16")
            b = mean_turns(cells, e, "mt-2k", "decode_tok_s", kv="q8")
            ch = "—" if not (a and b) else f"{(b / a - 1) * 100:+.1f}%"
            out.append(f"| {NAMES[e]} | {f(a)} | {f(b)} | {ch} |")
    return "\n".join(out) + "\n"


def write_csv(cells: dict, path: Path) -> None:
    fields = ["engine", "kv", "scenario", "turn", "rounds", "prompt_tokens", "cached_tokens", "ttft_s", "gen_s",
              "e2e_s", "decode_tok_s", "completion_tokens", "accept_rate", "peak_wired", "peak_rss", "needle_ok"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for (e, kv, s), c in sorted(cells.items(), key=lambda kv_: (CTX.index(kv_[0][2]), kv_[0][1], ENGINES.index(kv_[0][0]))):
            for t, d in c["turns"].items():
                w.writerow([e, kv, s, t, len(c["rounds"])] + [d.get(k) if k != "needle_ok" else
                                                                 ",".join(map(str, d[k])) for k in fields[5:]])


# ---------------------------------------------------------------- svg helpers


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_step(span: float, target: int = 6) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def svg_open(W, H, th, title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{esc(title)}" font-family="{FONT}">',
            f'<rect width="{W}" height="{H}" rx="8" fill="{th["bg"]}"/>',
            f'<text x="20" y="30" fill="{th["t1"]}" font-size="16" font-weight="600">{esc(title)}</text>',
            f'<text x="20" y="50" fill="{th["t2"]}" font-size="12">{esc(subtitle)}</text>']


def mark(o, cx, cy, color, th, kind, tip=None):
    t = f"<title>{esc(tip)}</title>" if tip else ""
    ring = f'stroke="{th["bg"]}" stroke-width="2"'
    if kind == "diamond":
        o.append(f'<path d="M{cx} {cy - 6}L{cx + 6} {cy}L{cx} {cy + 6}L{cx - 6} {cy}Z" fill="{color}" {ring}>{t}</path>')
    elif kind == "square":
        o.append(f'<rect x="{cx - 5}" y="{cy - 5}" width="10" height="10" rx="1.5" fill="{color}" {ring}>{t}</rect>')
    else:
        o.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}" {ring}>{t}</circle>')


def legend(o, th, engines, x, y, swatch=False):
    for e in engines:
        c = th[e]
        if swatch:
            o.append(f'<rect x="{x + 3}" y="{y - 10}" width="16" height="11" rx="2" fill="{c}"/>')
        else:
            o.append(f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" stroke="{c}" stroke-width="2"/>')
            mark(o, x + 11, y - 4, c, th, MARKER[e])
        o.append(f'<text x="{x + 28}" y="{y}" fill="{th["t2"]}" font-size="12">{esc(NAMES[e])}</text>')
        x += 40 + 7 * len(NAMES[e])


def line_chart(th, title, subtitle, ylabel, ctxs, series, *, logy=False, vfmt=lambda v: f"{v:.1f}", note=None):
    """series: {engine: [value or None per ctx]} — lines with markers, end labels (direct labels)."""
    W, H, L, R, T, B = 820, 440, 70, 130, 92, 64
    pw, ph = W - L - R, H - T - B
    vals = [v for s in series.values() for v in s if v]
    if logy:
        lo = 10 ** math.floor(math.log10(min(vals)))
        hi = 10 ** math.ceil(math.log10(max(vals)))
        fy = lambda v: T + ph * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
        ticks, t = [], lo
        while t <= hi * 1.0001:
            ticks.append(t)
            t *= 10
    else:
        step = nice_step(max(vals), 5)
        hi = step * math.ceil(max(vals) * 1.08 / step)
        lo = 0
        fy = lambda v: T + ph * (1 - v / hi)
        ticks = [i * step for i in range(int(round(hi / step)) + 1)]
    fx = lambda i: L + pw * (i + 0.5) / len(ctxs)
    o = svg_open(W, H, th, title, subtitle)
    legend(o, th, [e for e in ENGINES if e in series], 20, 76)
    for t in ticks:
        lab = (f"{t:g}" if t < 1 else (f"{t:,.0f}")) if logy else f"{t:g}"
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" stroke="{th["grid"]}"/>'
                 f'<text x="{L - 8}" y="{fy(t) + 4:.1f}" fill="{th["t2"]}" font-size="11" text-anchor="end">{lab}</text>')
    o.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="{th["axis"]}"/>')
    for i, c in enumerate(ctxs):
        o.append(f'<text x="{fx(i):.1f}" y="{T + ph + 18}" fill="{th["t2"]}" font-size="12" text-anchor="middle">'
                 f'{CTX_LABEL[c]}</text>')
    o.append(f'<text x="{L + pw / 2}" y="{H - 22}" fill="{th["t2"]}" font-size="11" text-anchor="middle">'
             f'prompt length (tokens)</text>')
    o.append(f'<text x="16" y="{T + ph / 2}" fill="{th["t2"]}" font-size="11" text-anchor="middle" '
             f'transform="rotate(-90 16 {T + ph / 2})">{esc(ylabel)}</text>')
    ends = []
    for e in ENGINES:
        if e not in series:
            continue
        pts = [(fx(i), fy(v), v) for i, v in enumerate(series[e]) if v]
        if not pts:
            continue
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)}" fill="none" '
                 f'stroke="{th[e]}" stroke-width="2" stroke-linejoin="round"/>')
        for (x, y, v), c in zip(pts, [c for c, v in zip(ctxs, series[e]) if v]):
            mark(o, round(x, 1), round(y, 1), th[e], th, MARKER[e], f"{NAMES[e]} · {CTX_LABEL[c]}: {vfmt(v)}")
        ends.append([pts[-1][1], e, pts[-1][2], pts[-1][0]])
    # end labels, de-collided
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 30)
    for y, e, v, x in ends:
        o.append(f'<text x="{x + 12:.1f}" y="{y - 1:.1f}" fill="{th["t1"]}" font-size="12" font-weight="600">'
                 f'{esc(vfmt(v))}</text><text x="{x + 12:.1f}" y="{y + 13:.1f}" fill="{th["t2"]}" font-size="11">'
                 f'{esc(NAMES[e])}</text>')
    if note:
        o.append(f'<text x="20" y="{H - 6}" fill="{th["t2"]}" font-size="11">{esc(note)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def conversation_chart(th, cells, ctxs):
    """Per context, one bar per engine: turns 1-3 end to end, TTFT solid + generation light.
    Each context gets its own time scale (small multiples), so short follow-up turns stay visible."""
    W, L, R, rowh, panel_gap = 900, 150, 90, 18, 58
    pw = W - L - R
    rows_per = [[e for e in ENGINES if (e, "q8", s) in cells] for s in ctxs]
    H = 100 + sum(len(r) * (rowh + 6) + panel_gap + 18 for r in rows_per) + 10
    o = svg_open(W, H, th, "A 3-turn conversation, end to end (8-bit KV)",
                 "Turn 1 reads the prompt cold; turns 2–3 add ~300 tokens each · ≤ 256 generated tokens per turn · "
                 "own time scale per panel")
    legend(o, th, ENGINES, 20, 76, swatch=True)
    o.append(f'<rect x="{W - 250}" y="66" width="16" height="11" rx="2" fill="{th["t2"]}"/>'
             f'<text x="{W - 228}" y="76" fill="{th["t2"]}" font-size="12">TTFT</text>'
             f'<rect x="{W - 180}" y="66" width="16" height="11" rx="2" fill="{th["t2"]}" fill-opacity="0.4"/>'
             f'<text x="{W - 158}" y="76" fill="{th["t2"]}" font-size="12">generation</text>')
    y = 104
    for s, engines in zip(ctxs, rows_per):
        top = max((conv_total(cells, e, s) or 0) for e in engines)
        unit, div = ("min", 60) if top > 150 else ("s", 1)
        step = nice_step(top / div, 5)
        hi = step * math.ceil(top / div / step)
        fx = lambda sec: L + pw * (sec / div) / hi
        o.append(f'<text x="20" y="{y + 6}" fill="{th["t1"]}" font-size="13" font-weight="600">'
                 f'{CTX_LABEL[s]} prompt</text>')
        y += 18
        ybot = y + len(engines) * (rowh + 6)
        t = 0.0
        while t <= hi + 1e-9:
            x = L + pw * t / hi
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y - 4}" y2="{ybot}" stroke="{th["grid"]}"/>'
                     f'<text x="{x:.1f}" y="{ybot + 14}" fill="{th["t2"]}" font-size="10" text-anchor="middle">'
                     f'{t:g} {unit if t == 0 else ""}</text>')
            t += step
        for e in engines:
            o.append(f'<text x="{L - 8}" y="{y + rowh - 5}" fill="{th["t2"]}" font-size="11" text-anchor="end">'
                     f'{esc(NAMES[e])}</text>')
            x = L
            for tn in (1, 2, 3):
                pre, gen = turn(cells, e, s, tn, "ttft_s"), turn(cells, e, s, tn, "gen_s")
                if pre is None or gen is None:
                    continue
                w1, w2 = max(fx(pre) - L, 1.5), max(fx(gen) - L, 1.5)
                tip = f"{NAMES[e]} {CTX_LABEL[s]} turn {tn}: TTFT {fs(pre)}, generation {fs(gen)}"
                o.append(f'<rect x="{x:.1f}" y="{y}" width="{w1:.1f}" height="{rowh}" rx="2" fill="{th[e]}">'
                         f'<title>{esc(tip)}</title></rect>')
                o.append(f'<rect x="{x + w1:.1f}" y="{y}" width="{w2:.1f}" height="{rowh}" rx="2" fill="{th[e]}" '
                         f'fill-opacity="0.4"><title>{esc(tip)}</title></rect>')
                x += w1 + w2 + 2  # 2px surface gap between turns
            o.append(f'<text x="{x + 6:.1f}" y="{y + rowh - 5}" fill="{th["t1"]}" font-size="11">'
                     f'{fs(conv_total(cells, e, s))}</text>')
            y += rowh + 6
        y += panel_gap
    o.append("</svg>")
    return "\n".join(o)


def charts(cells: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    ctxs = [s for s in CTX if any((e, "q8", s) in cells for e in ENGINES)]
    made = []
    for mode, th in THEMES.items():
        specs = {
            "decode": line_chart(
                th, "Decode speed vs prompt length (8-bit KV)",
                "Qwen3.8-27B, MTP depth 3, temperature 0 · mean of turns 1–3, median of rounds · higher is better",
                "decode tok/s", ctxs,
                {e: [mean_turns(cells, e, s, "decode_tok_s") for s in ctxs] for e in ENGINES}),
            "ttft-followup": line_chart(
                th, "Time to first token, turns 2–3 (follow-up)",
                "Previous turns should come from the engine's prefix cache · mean of turns 2 and 3 · log scale · lower is better",
                "seconds (log)", ctxs,
                {e: [mean_turns(cells, e, s, "ttft_s", ts=(2, 3)) for s in ctxs] for e in ENGINES}, logy=True, vfmt=fs),
            "memory": line_chart(
                th, "Peak wired memory during the conversation",
                "System-wide wired memory (vm_stat), max over turns 1–3 · 64 GB machine · lower is better", "GB", ctxs,
                {e: [(lambda v: v / GB if v else None)(max((turn(cells, e, s, t, "peak_wired") or 0) for t in (1, 2, 3)) or None)
                     for s in ctxs] for e in ENGINES}, vfmt=lambda v: f"{v:.1f} GB"),
            "conversation": conversation_chart(th, cells, ctxs),
        }
        for name, svg in specs.items():
            p = out / f"{name}-{mode}.svg"
            p.write_text(svg)
            made.append(p.name)
    return made


def main() -> None:
    run = Path(sys.argv[1])
    cells = build(load(run))
    (run / "summary.json").write_text(json.dumps({"|".join(k): v for k, v in cells.items()}, indent=1, default=list))
    write_csv(cells, run / "summary.csv")
    for lang in ("en", "ja"):
        (run / f"tables.{lang}.md").write_text(tables_md(cells, lang))
    made = charts(cells, run.parent.parent / "charts")
    print("cells:", len(cells), "charts:", len(made))


if __name__ == "__main__":
    main()

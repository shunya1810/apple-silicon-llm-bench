#!/usr/bin/env python3
""""Generate an SVG of a pelican riding a bicycle", thinking on, a few samples per engine.

Records time to first token (reasoning or answer), time to the first answer token,
reasoning and answer lengths, decode speed and total time, and saves each SVG.

  python scripts/pelican.py --engine engines/erp/splash-pelican.json --out results/pelican --n 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner import Engine, local_vars  # noqa: E402

PROMPT = "Generate an SVG of a pelican riding a bicycle"


def stream(base: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first = first_answer = last = None
    reasoning, answer = [], []
    usage = finish = error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    r = (d.get("reasoning_content") or "") + (d.get("reasoning") or "")
                    c = d.get("content") or ""
                    now = time.perf_counter() - t0
                    if r or c:
                        first = now if first is None else first
                        last = now
                    if r:
                        reasoning.append(r)
                    if c:
                        first_answer = now if first_answer is None else first_answer
                        answer.append(c)
                    finish = ch.get("finish_reason") or finish
    except Exception as exc:
        error = repr(exc)
    return {"total_s": time.perf_counter() - t0, "ttft_s": first, "first_answer_s": first_answer,
            "last_s": last, "reasoning": "".join(reasoning), "answer": "".join(answer),
            "usage": usage, "finish_reason": finish, "error": error}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--prompt-file", default=None, help="prompt text instead of the pelican prompt")
    ap.add_argument("--tag", default="pelican", help="name used in the output folder")
    a = ap.parse_args()
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else PROMPT
    spec = json.loads(Path(a.engine).read_text())
    out = Path(a.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cell = out / f"{stamp}-{a.tag}-{spec['id']}"
    cell.mkdir(parents=True)
    engine = Engine(spec, cell, local_vars())
    try:
        engine.start()
        warm = {"model": spec["model_id"], "messages": [{"role": "user", "content": "Say hello in five words."}],
                "max_tokens": 16, "temperature": 0.0, "stream": True}
        stream(engine.base, warm, 600)
        for i in range(1, a.n + 1):
            payload = {"model": spec["model_id"], "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                       "stream": True, "stream_options": {"include_usage": True}}
            payload.update(spec.get("erp_request", {}))
            r = stream(engine.base, payload, 10800)
            n = (r["usage"] or {}).get("completion_tokens")
            gen = (r["last_s"] - r["ttft_s"]) if r["ttft_s"] is not None and r["last_s"] else None
            m = re.search(r"<svg[\s\S]*?</svg>", r["answer"])
            code = re.search(r"```(?:python|py)\n([\s\S]*?)```", r["answer"])
            if code:
                (cell / f"{i}.py").write_text(code.group(1))
            (cell / f"{i}-reasoning.txt").write_text(r["reasoning"])
            (cell / f"{i}-answer.txt").write_text(r["answer"])
            if m:
                (cell / f"{i}.svg").write_text(m.group(0))
            row = {"engine": spec["id"], "tag": a.tag, "stamp": stamp, "i": i, "completion_tokens": n,
                   "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "code": bool(code),
                   "reasoning_chars": len(r["reasoning"]), "answer_chars": len(r["answer"]),
                   "ttft_s": r["ttft_s"], "first_answer_s": r["first_answer_s"], "total_s": r["total_s"],
                   "decode_tok_s": (n - 1) / gen if n and gen else None, "svg": bool(m),
                   "finish_reason": r["finish_reason"], "error": r["error"]}
            with (out / "rows.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

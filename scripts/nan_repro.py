#!/usr/bin/env python3
"""Reproduce the non-finite logits seen when decode crosses a compiled-verify bucket edge.

Reads a ~16.1K-token prompt, then generates up to 1,200 tokens so the context crosses
16,384 during decode. Prints finish_reason, completion tokens and any "non-finite" lines
from the server log.

  python scripts/nan_repro.py --engine engines/erp/fork.json [--env K=V ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner import Engine, build_context, expand, local_vars, stream_chat  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--context", type=int, default=16100)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "nan-repro"))
    a = ap.parse_args()
    spec = json.loads(Path(a.engine).read_text())
    spec["env"] = {**spec.get("env", {}), **dict(kv.split("=", 1) for kv in a.env)}
    if "--paged-kv-quantization" not in spec["cmd"]:
        spec["cmd"] = spec["cmd"] + spec.get("kv_variants", {}).get("q8", {}).get("cmd_extra", [])
    if "--max-tokens" in spec["cmd"]:
        spec["cmd"][spec["cmd"].index("--max-tokens") + 1] = "32768"
    cell = Path(a.out) / f"{time.strftime('%Y%m%d-%H%M%S')}-{spec['id']}"
    cell.mkdir(parents=True)
    from tokenizers import Tokenizer

    vars_ = local_vars()
    tok = Tokenizer.from_file(str(Path(expand(spec["tokenizer_dir"], vars_)) / "tokenizer.json"))
    ctx, n = build_context(tok, a.context, 200) if a.context else ("", 0)
    engine = Engine(spec, cell, vars_)
    try:
        engine.start()
        text = ctx + "\n\nWrite a long, detailed essay about these records." if a.context else "Generate an SVG of a pelican riding a bicycle"
        payload = {"model": spec["model_id"],
                   "messages": [{"role": "user", "content": text}],
                   "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": 0.95, "top_k": 20,
                   "stream": True, "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": not a.context}}
        if not a.context:
            payload["reasoning_effort"] = "xhigh"
        r = stream_chat(engine.base, payload, 3600)
        usage = r["usage"] or {}
        log = (cell / "server.log").read_text(errors="replace")
        bad = [l for l in log.splitlines() if "non-finite" in l or "NaN" in l]
        streamed = len(tok.encode(r["text"], add_special_tokens=False).ids)
        print(json.dumps({"env": a.env, "streamed_tokens": streamed, "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"), "finish": r["finish_reason"],
                          "error": r["error"], "nonfinite_lines": len(bad), "first": bad[:1]}), flush=True)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

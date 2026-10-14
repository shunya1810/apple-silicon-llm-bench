#!/usr/bin/env python3
"""Short prompts with thinking on (the Splash README conditions and a sampled variant), per engine.

  python scripts/think_short.py --engine engines/mtplx-fork-a3b-res.json --out results/x --n 2

Modes: "greedy1k" = temperature 0, max 1,024 tokens, reasoning medium (Splash's
docs/performance.md launch conditions); "sampled2k" = temperature 1.0, top_p 0.95,
top_k 20, max 2,048 tokens. Decode tok/s = (completion tokens - 1) / (last token
time - first token time), reasoning tokens included.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from runner import Engine, local_vars  # noqa: E402
from pelican import stream  # noqa: E402

PROMPTS = {
    "pelican": "Generate an SVG of a pelican riding a bicycle",
    "heptagon": (ROOT / "prompts/heptagon.txt").read_text(),
    "lru": "Write a thread-safe LRU cache in Python with get, put and a max size, plus unit tests.",
    "parse": "Write a Rust function that parses an ISO 8601 duration like P3DT4H5M6S into seconds, with tests.",
}
MODES = {
    "greedy1k": {"max_tokens": 1024, "temperature": 0.0},
    "sampled2k": {"max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
}
THINK = {
    "splash": {"reasoning_effort": "medium"},
    "mtplx": {"reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    spec = json.loads(Path(a.engine).read_text())
    # The engine's default KV variant, as orchestrate.py runs it (q8 pages /
    # Splash INT8): without it MTPLX starts with --paged-kv-quantization auto.
    variant = (spec.get("kv_variants") or {}).get(spec.get("default_kv") or "", {})
    spec["cmd"] = list(spec["cmd"]) + list(variant.get("cmd_extra", []))
    think = THINK["splash" if spec["model_id"] == "splash" else "mtplx"]
    out = Path(a.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cell = out / f"{stamp}-think-{spec['id']}"
    cell.mkdir(parents=True)
    engine = Engine(spec, cell, local_vars())
    try:
        engine.start()
        stream(engine.base, {"model": spec["model_id"], "messages": [{"role": "user", "content": "Say hello in five words."}],
                             "max_tokens": 16, "temperature": 0.0, "stream": True}, 600)
        for i in range(1, a.n + 1):
            for mode, sampling in MODES.items():
                for name, prompt in PROMPTS.items():
                    payload = {"model": spec["model_id"], "messages": [{"role": "user", "content": prompt}],
                               "stream": True, "stream_options": {"include_usage": True}, "seed": i, **sampling, **think}
                    r = stream(engine.base, payload, 3600)
                    n = (r["usage"] or {}).get("completion_tokens")
                    gen = (r["last_s"] - r["ttft_s"]) if r["ttft_s"] is not None and r["last_s"] else None
                    (cell / f"{mode}-{name}-{i}.txt").write_text(r["reasoning"] + "\n=====\n" + r["answer"])
                    row = {"engine": spec["id"], "stamp": stamp, "mode": mode, "prompt": name, "i": i,
                           "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "completion_tokens": n,
                           "reasoning_chars": len(r["reasoning"]), "answer_chars": len(r["answer"]),
                           "ttft_s": r["ttft_s"], "total_s": r["total_s"],
                           "decode_tok_s": (n - 1) / gen if n and gen else None,
                           "finish_reason": r["finish_reason"], "error": r["error"]}
                    with (out / "rows.jsonl").open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
